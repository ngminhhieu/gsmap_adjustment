import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

RESULT_DIR = "correlation/conv2d/default"

def plot_results(preds, gts, list_gauges=[1]):
	for i in list_gauges:
		gauge_gt = np.array(gts)[i-1,:]
		gauge_preds = np.array(preds)[i-1,:]  

		time = range(len(gauge_gt))
		plt.plot(time, gauge_preds, label="predicts")
		plt.plot(time, gauge_gt, label="groundtruths")
		plt.xlabel('precipitations')
		# Set the y axis label of the current axis.
		plt.ylabel('time')

		plt.title('Gauge {}'.format(i))
		# show a legend on the plot
		plt.legend()
		# Display a figure.
		plt.show()
		plt.savefig(RESULT_DIR + 'gauge_{}.png'.format(i))

def plot_training(training_history):
	epochs = np.array(training_history["epoch"])
	loss = np.array(training_history["loss"])
	val_loss = np.array(training_history["val_loss"])

	plt.plot(epochs, loss, label="loss")
	plt.plot(epochs, val_loss, label="val_loss")

	plt.xlabel('Loss')
	# Set the y axis label of the current axis.
	plt.ylabel('Epochs')

	plt.title('Training history')
	# show a legend on the plot
	plt.legend()
	# Display a figure.
	plt.show()
	plt.savefig(RESULT_DIR + 'training_history.png')

def metrics_analysis(df):
	df.columns = ["MAE", "MSE", "Margin"]
	df.index  = ["All"] + ["Gauge_{}".format(i+1) for i in range(72)]
	print("Worst MAE:\n", df.loc[df['MAE'].idxmax()])
	print("\nWorst MSE:\n", df.loc[df['MSE'].idxmax()])
	print("\nWorst Margin:\n", df.loc[df['Margin'].idxmax()])

if __name__ == '__main__':
	gt_df = pd.read_csv(os.path.join(RESULT_DIR,"groundtruth.csv"), header=None)
	preds_df = pd.read_csv(os.path.join(RESULT_DIR,"preds.csv"), header=None)
	metrics_df = pd.read_csv(os.path.join(RESULT_DIR,"list_metrics.csv"), header=None)
	metrics_analysis(metrics_df)
	plot_results(preds_df, gt_df, [1,2,3])
	history_df = pd.read_csv(os.path.join(RESULT_DIR,"training_history.csv"))
	# plot_training(history_df)
	
	