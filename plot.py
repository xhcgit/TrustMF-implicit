import pickle
import ToolScripts.Plotter as plotter
import matplotlib.pyplot as plt
import numpy as np
# from params import *

colors = ['red', 'cyan', 'blue', 'green', 'black', 'magenta', 'yellow', 'pink', 'purple', 'chocolate', 'orange', 'steelblue', 'crimson', 'lightgreen', 'salmon', 'gold', 'darkred']
lines = ['-', '--', '-.', ':']

def printBest(res):
	a = res['val_RMSE']
	index = list.index(a, min(a))
	print("test cv epoch = %d" %(index))
	print("best cv rmse = %.4f" % (a[index]))
	b = res['val_MAE']
	print("best cv mae = %.4f" % (b[index]))

	a = res['step_rmse']
	index = list.index(a, min(a))
	print("test best epoch = %d" %(index))
	print("best test rmse = %.4f" % (a[index]))
	b = res['step_mae']
	print("best test mae = %.4f" % (b[index]))

sets = [
	# "SR-GMI_dgi_Yelp_1586178650_CV1_rate0.8_r_0.01_hide_64_batch_128_batch2_128_LR_DACAY0.97_seed29",
	# "SR-GMI_dgi_Yelp_1586178638_CV1_rate0.8_r_0.01_hide_64_batch_256_batch2_256_LR_DACAY0.97_seed29",
	# "SR-GMI_dgi_Yelp_1586186079_CV1_rate0.8_r_0.01_hide_64_batch_64_batch2_64_LR_DACAY0.97_seed29",
	# "SR-GMI_dgi_Yelp_1586191612_CV1_rate0.8_r_0.005_hide_64_batch_128_batch2_128_LR_DACAY0.97_seed29",
	# "SR-GMI_dgi_Yelp_1586195055_CV1_rate0.8_r_0.005_hide_64_batch_64_batch2_64_LR_DACAY0.97_seed29",
	"SR-GMI_dgi_Yelp_1586215541_CV1_rate0.8_r_0.001_hide_64_batch_64_batch2_64_LR_DACAY0.97_seed29",
	# "SR-GMI_dgi_Yelp_1586215565_CV1_rate0.8_r_0.001_hide_64_batch_32_batch2_32_LR_DACAY0.97_seed29",
	"SR-GMI_dgi_Yelp_1586232928_CV1_rate0.8_r_0.001_hide_128_batch_64_batch2_64_LR_DACAY0.97_seed29",
	"SR-GMI_dgi_Yelp_1586232962_CV1_rate0.8_r_0.001_hide_32_batch_64_batch2_64_LR_DACAY0.97_seed29",
	"SR-GMI_dgi_Yelp_1586243041_CV1_rate0.8_r_0.0005_hide_64_batch_64_batch2_64_LR_DACAY0.97_seed29",
	# "SR-GMI_dgi_Yelp_1586255923_CV1_rate0.8_r_0.0001_hide_64_batch_64_batch2_64_LR_DACAY0.97_seed29"
]
names = [
	# "batch-128",#all_epoch_rmse=1.1614, all_epoch_mae=0.9147
	# "batch-256",
	# "batch-64-reg0.01",#all_epoch_rmse=1.1609, all_epoch_mae=0.9019
	# "batch-128-reg0.005",#all_epoch_rmse=1.1603, all_epoch_mae=0.9134
	# "batch-64-reg0.005",#all_epoch_rmse=1.1601, all_epoch_mae=0.9057
	"batch-64-hide-64-reg0.001",#all_epoch_rmse=1.1556, all_epoch_mae=0.9019
	# "batch-32-reg0.001",#all_epoch_rmse=1.1568, all_epoch_mae=0.9082
	"batch-64-hide-128-reg0.001",#all_epoch_rmse=1.1547, all_epoch_mae=0.9038
	"batch-64-hide-32-reg0.001",#all_epoch_rmse=1.1555, all_epoch_mae=0.8993
	"batch-64-hide-64-reg0.0005",#all_epoch_rmse=1.1554, all_epoch_mae=0.8916
	# "batch-64-hide-64-reg0.0001",#all_epoch_rmse=1.1637, all_epoch_mae=0.8994
]

sets = [
	"SR-GMI_dgi_CiaoDVD_1586454343_CV1_rate0.8_r_0.01_hide_64_u_batch_64_i_batch_256_t_batch_1024_LR_DACAY0.95",
	"SR-GMI_dgi_CiaoDVD_1586453939_CV1_rate0.8_r_0.01_hide_64_u_batch_64_i_batch_256_t_batch_1024_LR_DACAY0.95",
	"SR-GMI_dgi_CiaoDVD_1586454555_CV1_rate0.8_r_0.01_hide_64_u_batch_64_i_batch_256_t_batch_1024_LR_DACAY0.95",
	"SR-GMI_dgi_CiaoDVD_1586455061_CV1_rate0.8_r_0.01_hide_64_u_batch_64_i_batch_256_t_batch_1024_LR_DACAY0.95",
]
names = [
	"weight=True",
	"weight=False",
	"weight=True, no train dgi",
	"weight=False,  no train dgi",
]

sets = [
	# "SAMN_CiaoDVD_1586935801_CV1_rate0.8_r_0.001_hide_64_batch_64_batch_4_batch_4",
	# "SAMN_CiaoDVD_1586935803_CV1_rate0.8_r_0.005_hide_64_batch_64_batch_4_batch_4",
	# "SAMN_CiaoDVD_1586935806_CV1_rate0.8_r_0.01_hide_64_batch_64_batch_4_batch_4",
	"SAMN_CiaoDVD_1586938011_CV1_rate0.8_r_0.005_hide_64_batch_64_batch_4_batch_4",
	"SAMN_CiaoDVD_1586939178_CV1_rate0.8_r_0.01_hide_64_batch_64_batch_4_batch_4",
	"SAMN_CiaoDVD_1586939908_CV1_rate0.8_r_0.001_hide_64_batch_64_mem_4_att_4_lr_0.01"
]
names = [
	# "lr-0.001-reg-0.001",
	# "lr-0.001-reg-0.005",
	# "lr-0.001-reg-0.01",
	"lr-0.05-reg-0.001",
	"lr-0.05-reg-0.005",
	"lr-0.01-reg-0.001",
]
sets = [
	"SAMN_CiaoDVD_1586944868_CV1_rate0.8_r_0.005_hide_32_batch_64_mem_16_att_8_lr_0.05",
	"SAMN_CiaoDVD_1586944886_CV1_rate0.8_r_0.005_hide_32_batch_64_mem_16_att_16_lr_0.05",
]
names = [
	"16-8",
	"16-16"
]
smooth = 1
startLoc = 1
Length = 100

dataset = "CiaoDVD"
# dataset = "Yelp"

# assert len(names) == len(sets)
# a = sets[0]
# index1 = a.find('_') + 1
# b = a[index1:]
# index2 = b.find('_')
# dataset = b[0:index2]

for j in range(len(sets)):
	val = sets[j]
	name = names[j]
	print('val', val)
	with open(r'./History/' + dataset + r'/' + val + '.his', 'rb') as fs:
		res = pickle.load(fs)
	rmse = res['step_rmse']
	mae = res['step_mae']
	for i in range(len(rmse)):
		print("rmse %d: %.4f"%(i, rmse[i]))
	for i in range(len(mae)):
		print("mae %d: %.4f"%(i, mae[i]))
	
	# printBest(res)
	length = Length
	temy = [None] * 6
	temlength = len(res['loss'])
	temy[0] = np.array(res['loss'][startLoc: min(length, temlength)])
	temy[1] = np.array(res['RMSE'][startLoc: min(length, temlength)])
	temy[2] = np.array(res['val_loss'][startLoc: min(length, temlength)])
	temy[3] = np.array(res['step_rmse'][startLoc: min(length, temlength)])
	temy[4] = np.array(res['MAE'][startLoc: min(length, temlength)])
	temy[5] = np.array(res['step_mae'][startLoc: min(length, temlength)])
	for i in range(6):
		if len(temy[i]) < length-startLoc:
			temy[i] = np.array(list(temy[i]) + [temy[i][-1]] * (length-temlength))
	length -= 1
	y = [[], [], [], [], [], []]
	for i in range(int(length/smooth)):
		if i*smooth+smooth-1 >= len(temy[0]):
			break
		for k in range(6):
			temsum = 0.0
			for l in range(smooth):
				temsum += temy[k][i*smooth+l]
			y[k].append(temsum / smooth)
	y = np.array(y)
	length = y.shape[1]
	x = np.zeros((6, length))
	for i in range(6):
		x[i] = np.array(list(range(length)))
	plt.figure(1)
	plt.subplot(231)
	plt.title('LOSS FOR TRAIN')
	plt.plot(x[0], y[0], color=colors[j], label=name)
	plt.legend()

	plt.subplot(234)
	plt.title('LOSS FOR VAL')
	plt.plot(x[2], y[2], color=colors[j], label=name)
	plt.legend()

	plt.subplot(232)
	plt.title('RMSE FOR TRAIN')
	plt.plot(x[1], y[1], color=colors[j], label=name)
	plt.legend()

	plt.subplot(235)
	plt.title('RMSE FOR VAL')
	plt.plot(x[3], y[3], color=colors[j], label=name)
	plt.legend()

	plt.subplot(233)
	plt.title('MAE FOR TRAIN')
	plt.plot(x[4], y[4], color=colors[j], label=name)
	plt.legend()

	plt.subplot(236)
	plt.title('MAE FOR VAL')
	plt.plot(x[5], y[5], color=colors[j], label=name)
	plt.legend()

plt.show()
