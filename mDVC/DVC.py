import torch
from torch import nn

from subnet.layers import ResidualBlock, GDN, conv, deconv
from subnet.video_net import ME_Spynet, flow_warp, bilinearupsacling

from compressai.entropy_models import GaussianConditional, EntropyBottleneck

N = 128
M = 192

class DeepVideoCompressor(nn.Module):

    def __init__(self):
        super().__init__()

        # mv
        self.mv_encoder = nn.Sequential(
            nn.Conv2d(2, N, 3, stride=2, padding=1),
            GDN(N),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
            GDN(N),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
            GDN(N),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

        self.mv_decoder = nn.Sequential(
            nn.ConvTranspose2d(N, N, 3, stride=2, padding=1, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 3, stride=2, padding=1, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 3, stride=2, padding=1, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 2, 3, stride=2, padding=1, output_padding=1),
        )


        # mv.entropy model
        self.mv_prior_encoder, self.mv_prior_decoder = make_hyper_transform(N, M)
        self.mv_hyper_entropy_model = EntropyBottleneck(M)
        self.mv_conditionaL_entropy_model = GaussianConditional(None)


        # res
        self.res_encoder = nn.Sequential(
            nn.Conv2d(3, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
        )

        self.res_decoder = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1),
        )

        # res.entropy model
        self.res_prior_encoder, self.res_prior_decoder = make_hyper_transform(N, M)
        self.res_hyper_entropy_model = EntropyBottleneck(M)
        self.res_conditionaL_entropy_model = GaussianConditional(None)


        # motion
        self.opticFlow = ME_Spynet("./data/flow_pretrain_np/")
        self.refine = Refinement()


    def motion_compensation(self, ref, mv):
        warpped = flow_warp(ref, mv)
        feature = torch.cat((warpped, ref), 1)
        prediction = self.refine(feature) + warpped

        return prediction, warpped
    

    def forward(self, ref_frame, input_frame):
        # motion estimation
        flow = self.opticFlow(input_frame, ref_frame)   # [b, 2, h, w]

        # mv entropy
        flow_latent = self.mv_encoder(flow) # [b, 2, h/16, w/16]
        flow_latent_prior = self.mv_prior_encoder(flow_latent)

        q_flow_latent_prior, flow_latent_prior_likelihoods = self.mv_hyper_entropy_model(flow_latent_prior)

        gaussian_params = self.mv_prior_decoder(q_flow_latent_prior)
        flow_latent_scales, flow_latent_means = gaussian_params.chunk(2, 1) # mu, sigma

        q_flow_latent, flow_latent_likelihoods = self.mv_conditionaL_entropy_model(flow_latent, flow_latent_scales, means = flow_latent_means)

        q_flow = self.mv_decoder(q_flow_latent)

        # motion compensation
        prediction, warpped = self.motion_compensation(ref_frame, q_flow)

        # residual
        residual = input_frame - prediction
        res_latent = self.res_encoder(residual)

        # residual entropy
        res_latent_prior = self.res_prior_encoder(res_latent)
        q_res_latent_prior, res_latent_prior_likelihoods = self.res_hyper_entropy_model(res_latent_prior)

        gaussian_params = self.mv_prior_decoder(q_res_latent_prior)
        res_latent_scales, res_latent_means = gaussian_params.chunk(2, 1) # mu, sigma

        q_res_latent, res_latent_likelihoods = self.res_conditionaL_entropy_model(res_latent, res_latent_scales, means = res_latent_means)

        q_res = self.res_decoder(q_res_latent)

        recon_frame = prediction + q_res
        

        return recon_frame, {
            "p_mv_prior": flow_latent_prior_likelihoods,
            "p_mv": flow_latent_likelihoods,
            "p_res_prior": res_latent_prior_likelihoods,
            "p_res": res_latent_likelihoods
        }



    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)



def make_hyper_transform(N, M):
    h_a = nn.Sequential(
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, M),
            nn.LeakyReLU(inplace=True),
            conv(M, M),
        )
    
    h_s = nn.Sequential(
        deconv(M, N),
        nn.LeakyReLU(inplace=True),
        deconv(N, N * 3 // 2),
        nn.LeakyReLU(inplace=True),
        deconv(N * 3 // 2, N * 2, stride=1, kernel_size=3),
    )

    return h_a, h_s


class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()
        channelnum = 64

        self.feature_ext = nn.Conv2d(6, channelnum, 3, padding=1)# feature_ext
        self.f_relu = nn.ReLU()

        self.conv0 = ResidualBlock(channelnum, channelnum)#c0
        self.pooling0 = nn.AvgPool2d(2, 2)# c0p

        self.conv1 = ResidualBlock(channelnum, channelnum)#c1
        self.pooling1 = nn.AvgPool2d(2, 2)# c1p

        self.conv2 = ResidualBlock(channelnum, channelnum)# c2
        self.conv3 = ResidualBlock(channelnum, channelnum)# c3
        self.conv4 = ResidualBlock(channelnum, channelnum)# c4
        self.conv5 = ResidualBlock(channelnum, channelnum)# c5

        self.conv6 = nn.Conv2d(channelnum, 3, 3, padding=1)

        nn.init.xavier_uniform_(self.feature_ext.weight.data)
        nn.init.constant_(self.feature_ext.bias.data, 0.0)
        
        nn.init.xavier_uniform_(self.conv6.weight.data)
        nn.init.constant_(self.conv6.bias.data, 0.0)

    def forward(self, x):
        feature_ext = self.f_relu(self.feature_ext(x))

        c0 = self.conv0(feature_ext)
        c0_p = self.pooling0(c0)

        c1 = self.conv1(c0_p)
        c1_p = self.pooling1(c1)

        c2 = self.conv2(c1_p)
        c3 = self.conv3(c2)
        c3_u = c1 + bilinearupsacling(c3)

        c4 = self.conv4(c3_u)
        c4_u = c0 + bilinearupsacling(c4)

        c5 = self.conv5(c4_u)
        res = self.conv6(c5)
        return res


if __name__ == "__main__":
    from torchsummary import summary

    mDVC = DeepVideoCompressor()

    summary(mDVC, [(3, 256, 256), (3, 256, 256)], device = "cpu")