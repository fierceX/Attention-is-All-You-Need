#%%
import mxnet as mx
#%%
batch_size = 32
max_len = 100
embeding_size = 128
vocad_len = 252193


def Attention(data,Q_shape, K_shape, V_shape,name):
    Wq = mx.sym.var(name=('%s_Wq' % name),shape=Q_shape,init=mx.init.Normal())
    Wk = mx.sym.var(name=('%s_Wk' % name),shape=K_shape,init=mx.init.Normal())
    Wv = mx.sym.var(name=('%s_Wv' % name),shape=V_shape,init=mx.init.Normal())

    data_wq = mx.sym.dot(data,Wq)
    data_wk = mx.sym.dot(data,Wk)
    data_wv = mx.sym.dot(data,Wv)

    S = mx.sym.batch_dot(data_wq, mx.sym.transpose(data_wk, axes=(0, 2, 1))) / (Q_shape[1]**0.5)
    return mx.sym.batch_dot(mx.sym.softmax(S), data_wv)

def Multi_Head_Attention(data,Q_shape, K_shape, V_shape, h,name):
    Wo = mx.sym.var(name=('%s_Wo' % name), shape=(h * V_shape[1], V_shape[0]),init=mx.init.Normal())

    H = []
    for i in range(h):
        H.append(Attention(data,Q_shape,K_shape,V_shape,name=('%s_%s'%(name,i))))
    return mx.sym.dot(mx.sym.concat(*H, dim=2), Wo)

qkv_shape=(embeding_size,max_len)
data = mx.sym.var('data')
embe = mx.sym.Embedding(data=data,input_dim=vocad_len,output_dim=embeding_size)

MHA = Multi_Head_Attention(data=embe,Q_shape=qkv_shape,K_shape=qkv_shape,V_shape=qkv_shape,h=8,name='HMA')
pool = mx.sym.Pooling(data=MHA,global_pool=True,pool_type='avg',kernel=1,name='pool')
liner = mx.sym.FullyConnected(data=pool,num_hidden=2,name='FC')
net = mx.sym.SoftmaxOutput(data=liner, name='softmax')

#%%
net.infer_shape(data=(32,100))