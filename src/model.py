import torch
import torch.nn as nn

class ConvLSTM_Cell(nn.Module):
    def __init__(self,input_dim,hidden_dim,kernel_size,bias=True):
        super(ConvLSTM_Cell,self).__init__()

        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.kernel_size=kernel_size

        padding=kernel_size//2

        self.conv=nn.Conv2d(
            in_channels=self.input_dim+self.hidden_dim,
            out_channels=4*self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=bias
            )

    def forward(self,input_tensor,curr_state):
        h_cur,c_cur=curr_state

        combined=torch.cat([input_tensor,h_cur],dim=1)
        combined_conv=self.conv(combined)

        cc_i,cc_f,cc_o,cc_g=torch.split(combined_conv,self.hidden_dim,dim=1)
        i=torch.sigmoid(cc_i)
        f=torch.sigmoid(cc_f)
        o=torch.sigmoid(cc_o)
        g=torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next) 

        return h_next,c_next

    def init_hidden(self,batch_size,image_size):
        #initializing empty memory states
        height,width=image_size
        device=self.conv.weight.device
        return (torch.zeros(batch_size,self.hidden_dim,height,width,device=device),
                torch.zeros(batch_size,self.hidden_dim,height,width,device=device))


class OceanForecaster(nn.Module):
    def __init__(self,input_channels=7,hidden_channels=16,output_channels=15,kernel_size=3):
        super(OceanForecaster,self).__init__()

        self.hidden_channels=hidden_channels

        self.conv_lstm=ConvLSTM_Cell(input_channels,self.hidden_channels,kernel_size=kernel_size)

        self.final_conv=nn.Conv2d(in_channels=hidden_channels,out_channels=output_channels,kernel_size=1)

    def forward(self,x):
        b, seq_len, _, h, w = x.size()    # x shape: (Batch_Size, Time_Steps, Channels=7, Height, Width)

        h_t,c_t=self.conv_lstm.init_hidden(batch_size=b,image_size=(h,w))

        for t in range(seq_len):
            h_t,c_t=self.conv_lstm(x[:, t, :, :, :],(h_t,c_t))

        output=self.final_conv(h_t)

        return output


if __name__ == "__main__":


    dummy_input = torch.randn(4, 7, 7, 101, 241) 

    model = OceanForecaster(input_channels=7, hidden_channels=16, output_channels=15)

    predictions = model(dummy_input)

    print("SUCCESS! Model compiled and executed.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {predictions.shape} -> Expected: (4, 15, 101, 241)")