import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

maxlen = 2**20
padding_char = 256

def get_sample(path):
    if not os.path.exists(path):
        raise FileNotFoundError("The file doesn't exist")
    else:
        with open(path, 'rb') as f:
            prog = f.read()
            b = np.ones((maxlen,), dtype=np.uint16) * padding_char
            bytez = np.frombuffer(prog[:maxlen], dtype=np.uint8)
            b[:len(bytez)] = bytez 
            return b

def get_min_wrt(d, s):
	d = np.array(d)
	min_idx = d.argsort()
	for x in min_idx:
		if s[x] > 0:
			return x
	return 0


model = load_model("malconv.h5")
x = get_sample("PE/Backdoor3.exe")

session = K.get_session()
grads = K.gradients(model.output, model.layers[1].output)


I = np.arange(2, int(0x3c))
T = 10
t = 0
M = np.array(np.arange(0,2**20))
M[255:] = 0
M_emb = session.run(model.layers[1].output, feed_dict={model.input: [M]})[0]
x_0 = np.copy(x)
pred = model.predict(np.asarray([x]))

while(pred > 0.5 and t < T):
	# print(x_0)
	Z = session.run(model.layers[1].output, feed_dict={model.input: [x_0]})[0]
	# print("====================EMBEDDED LAYER====================")
	# print(Z)
	adv_x = []
	grads_ = session.run(grads, feed_dict={model.input: [x_0]})
	grads_ = -grads_[0][0]
	# print("====================GRADIENTS====================")
	# print(grads_)
	for i in I:
		g = grads_[i]/np.linalg.norm(grads_[i], 2)
		s = []
		d = []
		for b in range(0,256):
			z_b = M_emb[b]
			s.append(np.dot(g.T,(z_b - Z[i])))
			d.append(np.linalg.norm(z_b - (Z[i] + s[b]*g)))
			
		adv_x.append(get_min_wrt(d,s))
	x_0[2:int(0x3c)] = adv_x
	# print("====================ADVERSARIAL BYTES====================")
	# print(adv_x)
	pred = model.predict(np.asarray([x_0]))
	print(pred)
	print("Iteration n: {}".format(t))
	# print(x[:int(0x3c)])
	t = t + 1

# print(adv_x)


print("BEFORE")
print(x[:61])
print(model.predict(np.asarray([x])))

print("AFTER")
print(x_0[:61])
print(model.predict(np.asarray([x_0])))

