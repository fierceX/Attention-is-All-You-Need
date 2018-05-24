#%%
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn


#%%
def SacledDotProductAttention(q, k, v, ctx=mx.cpu()):
    S = nd.batch_dot(q, nd.transpose(k, axes=(0, 2, 1))) / nd.sqrt(
        nd.array([q.shape[1]], ctx=ctx))
    return nd.batch_dot(nd.softmax(S), v)


#%%
class Multi_Head_Attention(nn.Block):
    def __init__(self, Q_shape, K_shape, V_shape, h, ctx=mx.cpu(), **kwargs):
        super(Multi_Head_Attention, self).__init__(**kwargs)
        self.Wo = self.params.get('Wo', shape=(h * V_shape[1], V_shape[0]))
        self.Q_shape = Q_shape
        self.K_shape = K_shape
        self.V_shape = V_shape
        self.h = h
        self.ctx = ctx
        self.Wq = []
        self.Wk = []
        self.Wv = []
        self.h = h
        for i in range(h):
            self.Wq.append(self.params.get('Wq' + str(i), shape=Q_shape))
            self.Wk.append(self.params.get('Wk' + str(i), shape=K_shape))
            self.Wv.append(self.params.get('Wv' + str(i), shape=V_shape))

    def forward(self, q, k, v):
        H = []
        for i in range(self.h):
            H.append(
                SacledDotProductAttention(
                    nd.dot(q, self.Wq[i].data()),
                    nd.dot(k, self.Wk[i].data()),
                    nd.dot(v, self.Wv[i].data()),
                    ctx=self.ctx))
        return nd.dot(nd.concat(*H, dim=2), self.Wo.data())

    def __repr__(self):
        s = '{name}(Q_shape={Q_shape},K_shape={K_shape},V_shape={V_shape},out_shape={out_shape},h_num={h})'
        return s.format(
            name=self.__class__.__name__,
            Q_shape=str((self.Q_shape[1], self.Q_shape[0])),
            K_shape=str((self.K_shape[1], self.K_shape[0])),
            V_shape=str((self.V_shape[1], self.V_shape[0])),
            out_shape=str((self.K_shape[1], self.V_shape[0])),
            h=str(self.h))


class SANet(gluon.Block):
    def __init__(self, shape, Vocad_len, h, ctx=mx.cpu(), **kwargs):
        super(SANet, self).__init__(**kwargs)
        self.embed = nn.Embedding(input_dim=Vocad_len, output_dim=shape[0])
        self.MHA = Multi_Head_Attention(
            Q_shape=shape, K_shape=shape, V_shape=shape, h=h, ctx=ctx)
        self.liner = gluon.nn.Dense(2, activation='relu')
        self.pool = gluon.nn.GlobalAvgPool1D()

    def forward(self, x):
        kqv = self.embed(x)
        return self.liner(self.pool(self.MHA(kqv, kqv, kqv)))
