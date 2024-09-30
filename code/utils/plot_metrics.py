import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


met_df: pd.DataFrame = pd.read_pickle("metrics.pkl");

met_df = met_df.assign(epoch=met_df.index+1);

# Fix incorrect metric reporting in from old code
print(met_df.columns.to_list())
if ("val_mse" in met_df.columns.to_list()): met_df = met_df.rename( columns= { "val_mse": "val_mae" } );


# met_df["epoch"] += 99;
print(met_df);

met_mdf = met_df.melt("epoch", var_name="metric", value_name="value")

sns.set_theme();
sns.regplot(data=met_df.where(met_df.epoch > 10), x="epoch", y="val_mae", ax=plt.subplot(2,2,1));
sns.regplot(data=met_df.where(met_df.epoch > 10), x="epoch", y="train_loss", ax=plt.subplot(2,2,2));
sns.lineplot(data=met_mdf.where(met_mdf.epoch > 10), x="epoch", y="value", hue="metric", palette="flare", ax=plt.subplot(2,2,3));
sns.lineplot(data=met_mdf.where(met_mdf.epoch < 10), x="epoch", y="value", hue="metric", palette="flare", ax=plt.subplot(2,2,4));
plt.show();