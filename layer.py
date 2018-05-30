#%%
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn


#%%
class Attention(nn.HybridBlock):
    def __init__(self, Q_shape, K_shape, V_shape, **kwargs):
        super(Attention, self).__init__(**kwargs)
        with self.name_scope():
            self.Wq = self.params.get('Wq', shape=Q_shape)
            self.Wk = self.params.get('Wk', shape=Q_shape)
            self.Wv = self.params.get('Wv', shape=Q_shape)
            self.qq = Q_shape[1]
        self.Q_shape = Q_shape
        self.K_shape = K_shape
        self.V_shape = V_shape

    def SacledDotProductAttention(self, F, q, k, v):
        S = F.batch_dot(q, F.transpose(k, axes=(0, 2, 1))) / (self.qq**0.5)
        return F.batch_dot(F.softmax(S), v)

    def hybrid_forward(self, F, q, k, v, Wq, Wk, Wv):
        return self.SacledDotProductAttention(F, F.dot(q, Wq), F.dot(k, Wk),
                                              F.dot(v, Wv))

    def __repr__(self):
        s = '{name}(Q_shape={Q_shape}, K_shape={K_shape}, V_shape={V_shape}, out_shape={out_shape})'
        return s.format(
            name=self.__class__.__name__,
            Q_shape=str((self.Q_shape[1], self.Q_shape[0])),
            K_shape=str((self.K_shape[1], self.K_shape[0])),
            V_shape=str((self.V_shape[1], self.V_shape[0])),
            out_shape=str((self.K_shape[1], self.V_shape[0])))


class Position_Embedding(nn.HybridBlock):
    def __init__(self, shape, **kwargs):
        super(Position_Embedding, self).__init__(**kwargs)
        self.shape = shape

    def hybrid_forward(self, F, x):
        seq_len = self.shape[0]
        position_size = self.shape[1]
        position_j = 1. / F.broadcast_power(
            10000 * F.ones(shape=(1, 1)),
            (2 * F.arange(position_size / 2) / position_size))
        position_i = F.arange(seq_len, dtype='float32')
        position_i = F.expand_dims(position_i, 1)
        position_ij = F.dot(position_i, position_j)
        position_ij = F.concat(
            *[F.cos(position_ij), F.sin(position_ij)], dim=1)
        position_ij = F.broadcast_add(
            F.expand_dims(position_ij, 0), F.zeros_like(x))

        return position_ij + x


class Multi_Head_Attention(nn.HybridBlock):
    def __init__(self, Q_shape, K_shape, V_shape, h, **kwargs):
        super(Multi_Head_Attention, self).__init__(**kwargs)
        with self.name_scope():
            for _ in range(h):
                self.register_child(
                    Attention(
                        Q_shape=Q_shape,
                        K_shape=K_shape,
                        V_shape=V_shape))
            self.Wo = self.params.get('Wo', shape=(h * V_shape[1], V_shape[0]))
        self.h = h

    def hybrid_forward(self, F, q, k, v, Wo):
        H = []
        for block in self._children.values():
            H.append(block(q, k, v))
        return F.dot(F.concat(*H, dim=2), Wo)

    def __repr__(self):
        s = '{name}({Attention}, h_num={h})'
        return s.format(
            name=self.__class__.__name__,
            Attention=list(self._children.values())[0],
            h=self.h)


class SANet(gluon.HybridBlock):
    def __init__(self, shape, Vocad_len, h,Is_PE=True, **kwargs):
        super(SANet, self).__init__(**kwargs)
        self.embed = nn.Embedding(input_dim=Vocad_len, output_dim=shape[0])
        self.PE = Position_Embedding(shape=(shape[1], shape[0]))
        self.MHA = Multi_Head_Attention(
            Q_shape=shape, K_shape=shape, V_shape=shape, h=h)
        self.liner = gluon.nn.Dense(2)
        self.pool = gluon.nn.GlobalAvgPool1D()
        self.droup = gluon.nn.Dropout(.5)
        self.shape = (shape[1],shape[0])
        self.Is_PE = Is_PE
    
    def hybrid_forward(self, F, x):
        kqv = self.PE(self.embed(x))
        kqv = self.embed(x)
        if self.Is_PE:
            kqv = self.PE(kqv)
        return self.liner(self.droup(self.pool(self.MHA(kqv, kqv, kqv))))
