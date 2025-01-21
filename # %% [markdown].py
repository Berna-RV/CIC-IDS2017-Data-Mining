# %% [markdown]
# ### Concatenate the dataset csv's

# %%
import pandas as pd
import os

def concat_csv_files(folder_path):
    """
    Concatenates all CSV files within a specified folder into a single DataFrame.

    Args:
        folder_path: The path to the folder containing the CSV files.

    Returns:
        A pandas DataFrame containing the concatenated data, or None if no CSV 
        files are found or an error occurs.
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not all_files:
        print("No CSV files found in the specified folder.")
        return None
    try:
        df_list = []
        for filename in all_files:
          file_path = os.path.join(folder_path, filename)
          df = pd.read_csv(file_path, low_memory=False)
          df_list.append(df)
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)
        return combined_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


folder_path = "CIC-IDS2017"
combined_dataframe = concat_csv_files(folder_path)

if combined_dataframe is not None:
    print("CSV files concatenated successfully.")
    combined_dataframe.to_csv("cicids2017.csv", index=False)

# %% [markdown]
# ## Preprocessing
# 
# ### Feature reduction
# 
# According to the paper the features to drop (because they are redundant or meaningless) we drop 
# six features, namely, “Flow ID”, “Source IP ”, “Source Port”, “Destination IP”, “Protocol”, and “Time stamp”, hence reducing its feature dimension to 77. (This features were already removed from the downloaded dataset).
# 
# ### Feature selection
# In this section I analyse the motives that should imply why a columns should be removed from the dataset.
# 
# #### Features to consider removing
# 
# ##### Highly Correlated Features:
# + Features that are highly correlated (e.g., Fwd Packet Length Max and Fwd Packet Length Mean) may provide redundant information. The usage of a correlation matrix to identify these is a good method.
# 
# ##### Statistical Aggregates:
# + Detailed statistics (mean, max, min, std) for packet lengths, I considered keeping only a few representative aggregates. For example:
#   + Keeping Mean or Std and drop Max and Min if their patterns do not add unique predictive power.
# 
# ##### Low Variance Features:
# + Features that show little or no variation across the dataset do not contribute to distinguishing between classes. 
#   + Example: Flags like Fwd PSH Flags, Bwd URG Flags, CWE Flag Count.
# 
# In the next code block I do some feature selection in a simple way, but in the section "More Feature Selection" a deeper analysis towards this goal in made.
# 
# #### Dataset cleaning
# 
# Here we make all features to be greater or equal than 0, identify and remove the columns that have zero variance (feature selection due to low variance in this features as explained previously), drop the rows that have infinite and nan values, drop the duplicates and finally we drop the columns we identical values (feature selection due to a great correlation between this features as explained previously).

# %%
import pandas as pd
import numpy as np
from itertools import combinations, product

def clean_df(df):
    # Remove the space before each feature names
    df.columns = df.columns.str.strip()
    print('dataset shape', df.shape)

    # This set of feature should have >= 0 values
    num = df._get_numeric_data()
    num[num < 0] = 0

    zero_variance_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            zero_variance_cols.append(col)
    df.drop(zero_variance_cols, axis = 1, inplace = True)
    print('zero variance columns', zero_variance_cols, 'dropped')
    print('shape after removing zero variance columns:', df.shape)

    df.replace([np.inf, -np.inf], np.nan, inplace = True)
    print(df.isna().any(axis = 1).sum(), 'rows dropped')
    df.dropna(inplace = True)
    print('shape after removing nan:', df.shape)

    # Drop duplicate rows
    df.drop_duplicates(inplace = True)
    print('shape after dropping duplicates:', df.shape)

    column_pairs = [(i, j) for i, j in combinations(df, 2) if df[i].equals(df[j])]
    ide_cols = []
    for column_pair in column_pairs:
        ide_cols.append(column_pair[1])
    df.drop(ide_cols, axis = 1, inplace = True)
    print('columns which have identical values', column_pairs, 'dropped')
    print('shape after removing identical value columns:', df.shape)
    return df

# Load CIC-IDS2017 dataset
df = pd.read_csv("cicids2017.csv")
df = clean_df(df)
df['Label'].value_counts()
df.to_csv('step1_cleaned_dataset_cicids2017.csv', index=False)

# %%
df['Label'].value_counts()

# %% [markdown]
# Here we can see that some of the labels have non readable characters, so we will change their value.

# %%
### rename Labels that contain non-printable characters 
print("Before rename...")
print(df.loc[:,"Label"].unique())

df.loc[:,"Label"].replace({"Web Attack � XSS" : "XSS", "Web Attack � Sql Injection": "Sql Injection", "Web Attack � Brute Force": "Brute Force"}, inplace=True)
print("After rename..")
print(df.loc[:,"Label"].unique())

## remove trailing && leading spaces from all the labels
rename_cols = lambda col_lbl: col_lbl.strip()
df.rename(rename_cols, axis=1, inplace=True, errors="raise")

df.to_csv('step2_labels_renamed_cicids2017.csv')

df['Label'].value_counts()

# %% [markdown]
# #### Data visualization
# 
# Data visualization can help us identify what features are more likelly to make the difference in the predicting task.

# %%
# temporarily add new column to distinguish traffic type between Normal / Attack 

trafic_type = df.loc[:, "Label"].map(lambda lbl: "Normal" if lbl == "BENIGN" else "Attack")
trafic_type.name = "Traffic type"
df.loc[:, trafic_type.name] = trafic_type

print(df.shape)

# %% [markdown]
# #### Plot distribution of Normal traffic and Attacks

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.ticklabel_format(axis='y', useMathText=True, useOffset=False)

sns.countplot(x="Traffic type", data=df, palette=["g","r"])

plt.title("Traffic type distribution")
plt.xlabel("Traffic Type")
plt.ylabel("Number of instances")
# plt.savefig("distribution1.png", dpi=200, format='png')
plt.show()

# %% [markdown]
# As we can see the dataset is heavily imbalanced, the Normal traffic outweights the Malicious traffic in great manner. With a dataset like this the model might get biased one class, to avoid this we can under-sample or over-sample.
# 
# Do to hardware requisitions when training a model with a big dataset, I will Under-sample (this will disminuish the number of observations in the dataset) in this case, but keeping in mind that Over-sampling might be the best option in this kind of task.

# %%
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=10, sampling_strategy=0.85) # equals traffic unless ratio is specified

#df.drop(["Traffic type"], axis=1, inplace=True) # temporarily remove the categorical column for underSampling

df_data_res, traffic_type_res = rus.fit_resample(df, trafic_type)

dfv2 = df_data_res.join(traffic_type_res, how="inner")

dfv2.shape

# %%
### show distribution chart after downsampling Normal traffic 

plt.ticklabel_format(axis='y', useMathText=True, useOffset=False)  # change def ScalarFormatter
sns.countplot(x="Traffic type", data=dfv2, order=["Normal", "Attack"],  palette=["g","r"])

#plt.title("Traffic type distribution in whole dataset after random downsampling")
plt.title("Traffic type distribution after random downsampling")
plt.xlabel("Traffic Type")
plt.ylabel("Number of instances")
# plt.savefig("distribution2.png", dpi=200, format='png')
plt.show()

# %% [markdown]
# ### More Feature Selection

# %%
labels = dfv2.loc[:, "Label"] # labels column
features = dfv2.iloc[:, :-2] # removing the "Label" and "Traffic type" columns 

features.info()

# %% [markdown]
# I'll use RandomForestClassifier to check the attributes importance.

# %%
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=10, n_jobs=-1) # 100 trees in forest

# fit random forest classifier on the dataset
rfc.fit(features, labels)

score = np.round(rfc.feature_importances_,5)

importances = pd.DataFrame({'Characteristics': features.columns,'Importance level': score})
importances = importances.sort_values('Importance level',ascending=False).set_index('Characteristics')

# plot importances
sns.barplot(x=importances.index, y="Importance level", data=importances, color="b")
plt.xticks(rotation="vertical")
plt.gcf().set_size_inches(14,5)
# plt.savefig("importances.png", dpi=200, format='png', bbox_inches = "tight", pad_inches=0.2)
plt.show()



# %% [markdown]
# Here I will create a threshold that if the importance score is smaller the column in question will not be considered.

# %%
threshold = 0.001 # importance threshold

bl_thresh = importances.loc[importances["Importance level"] < threshold]
print("There are {} features to delete, as they are below the chosen threshold".format(bl_thresh.shape[0]))
print("These features are the following:")
feats_to_del = [feat for feat in bl_thresh.index]
print("\n".join(feats_to_del))

# removing these not important features 
dfv2.drop(columns=feats_to_del, inplace=True) # dropping columns

dfv2

# %% [markdown]
# #### Detect higly correlated pairs visually

# %%
df_correlation_matrix = dfv2.iloc[:, :-2].corr() # removing the "Label" and "Traffic type" columns 

plt.gcf().set_size_inches(60, 60)
hm = sns.heatmap(df_correlation_matrix, annot=True, linewidths=.8, annot_kws={"fontsize": 15}, fmt=".2f")
hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize = 25)
hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize = 25)
# plt.savefig("corr_heatmap.png", dpi=200, format='png', bbox_inches = "tight", pad_inches=0.4)
plt.show()

# %%
df_correlation_matrix

# %% [markdown]
# Listing of highly correlated pairs:

# %%
def srt_corr(mtrx):
    # Unstack the correlation matrix
    corr_ustack = mtrx.unstack().abs()
    # Remove self-correlations by filtering out entries where the row and column indices are the same
    corr_ustack = corr_ustack[corr_ustack.index.get_level_values(0) != corr_ustack.index.get_level_values(1)]
    # Sort the correlations
    corr_srted = corr_ustack.sort_values(ascending=False)
    
    return corr_srted

srt_corr(df_correlation_matrix)

# %%
from collections import OrderedDict
thres_corr = 0.95
epoch=0
highly_corr = {"dummy": "dummy"}
feats_deled = []

all_data_corr_mtrx2 = df_correlation_matrix.copy()
def add_to_dct(l, ft, ft2):
    try:
        l[ft].append(ft2)
    except KeyError:
        l[ft] = [ft2]

get_imp = lambda feat: importances.loc[feat][0]
srt_key = lambda elem: get_imp(elem[0])  # gets importance of first element

def what_to_del(dct_srt):    
    to_del = []  # least important feature
    for k, val in dct_srt.items():
        # get all indexes lower than current k
        feats_lw_imp = importances[importances.index.slice_indexer(k)].index
        if set(val) - set(feats_lw_imp):  # feature k creates a correlation pair w/ feature of higher importance --- delete feat k
            if k not in to_del: to_del.append(k)
        else:  # feature k creates a correlation pair w/ features of lower importance --- delete one w/ lowest importance 
            for ft in feats_lw_imp[::-1]:  # searching from least important
                if ft in val and ft not in to_del:
                    to_del.append(ft)
                    break  # deleting first foundend feature of lowest possible importance
    return to_del
        
while highly_corr:
    count = 0
    highly_corr.clear()
    for feats, val in srt_corr(all_data_corr_mtrx2).items():
        if val > thres_corr and feats[0] != feats[1]:
            count += 1
            add_to_dct(highly_corr, feats[0], feats[1])
    if not highly_corr: break  # no more highly correlated pairs
    highly_corr_srt = OrderedDict(sorted(highly_corr.items(), key=srt_key))  # sorted based on importance

    to_del = what_to_del(highly_corr_srt)
    feats_deled += to_del
    epoch +=1 # first epoch will be 1 not 0! 
    print("There are {} higly correlated pairs in {} iteration".format(count, epoch))
    all_data_corr_mtrx2.drop(to_del, axis=1, inplace=True)
    all_data_corr_mtrx2.drop(to_del, axis=0, inplace=True)  # need to remove the feat from both cols and index

print("Deleting: {} feature".format(len(feats_deled)))
print("Finally deleted:\n"+ "\n".join(feats_deled))

# %% [markdown]
# #### Plotting correlation matrix heatmap after removing highly correlated pairs

# %%
plt.gcf().set_size_inches(40, 40)

# Plot the heatmap
hm2 = sns.heatmap(all_data_corr_mtrx2, annot=True, linewidths=.8, annot_kws={"fontsize": 15}, fmt=".2f")

# Adjust tick labels directly from hm2
hm2.set_yticklabels(hm2.get_yticklabels(), fontsize=20)
hm2.set_xticklabels(hm2.get_xticklabels(), fontsize=20)

# Save and show the plot
# plt.savefig("corr_heatmap2.png", dpi=200, format='png', bbox_inches="tight", pad_inches=0.4)
plt.show()

# %%
dfv3 = dfv2.copy()

dfv3.drop(feats_deled, axis=1, inplace=True)

dfv3_cp = dfv3.copy()

dfv3.to_csv('feature_selected_cicids2017.csv')

dfv3

# %%
from sklearn.decomposition import PCA
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# dfv3 = pd.read_csv('feature_selected_cicids2017.csv')

subsample_df = dfv3.groupby('Label').apply(pd.DataFrame.sample, frac = 0.1).reset_index(drop = True)

X = subsample_df.drop(['Label', 'Traffic type'], axis = 1)
y = subsample_df['Label']

pca = PCA(n_components = 2, random_state = 0)
z = pca.fit_transform(X) 

pca_15_df = pd.DataFrame()
pca_15_df['Label'] = y
pca_15_df['dimension 1'] = z[:, 0]
pca_15_df['dimension 2'] = z[:, 1]

sns.scatterplot(x = 'dimension 1', y = 'dimension 2', 
                hue = pca_15_df.Label,
                palette = sns.color_palette('hls', len(pca_15_df.Label.value_counts())),
                data = pca_15_df).set(title = 'CICIDS2017 15 Classes PCA Projection')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5)) 
plt.show()

# %% [markdown]
# ### Data standardization
# 
# Standardize the data to a Gaussian distribution with a mean of 0 and variance of 1:
# 
# $$x' = \frac{x - \mu}{\delta}$$
# 
# Where x is the original feature, x ′ is the normalized feature, and 𝜇 and 𝛿 are the mean and standard deviation of the feature, respectively. The normalized data maintains the same linear relationship as the original data.
# 
# 
# 

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

dfv3 = pd.read_csv('feature_selected_cicids2017.csv')

# scaler = StandardScaler()

qt = QuantileTransformer(random_state=10) # number of quantiles can be set, default n_quantiles=1000

labels = dfv3.loc[:, "Label"]

binary_labels = dfv3.loc[:, "Traffic type"]

dfv3.drop(["Label", "Traffic type"], axis=1, inplace=True) # drop categorical columns

# dfv3_scalled = scaler.fit_transform(dfv3)
dfv3_scalled = qt.fit_transform(dfv3)

dfv3_scalled

# %% [markdown]
# ### Train Test Split

# %%
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels  = train_test_split(dfv3_scalled, labels, random_state=10, train_size=0.7) # 70/30 train test split


# %% [markdown]
# ### Test Validation Split

# %%
train_features, validation_features, train_labels, validation_labels = train_test_split(train_features, train_labels, random_state=10, train_size=0.8)

# %%
labels_count = train_labels.value_counts()
all_samples = labels_count.sum()
print(labels_count)
print("Total: {}".format(all_samples))

# %% [markdown]
# ### Classes distribution

# %%
order = labels_count.index
palette = {}
for key in order:
    palette[key] = "g" if key == "BENIGN" else "r"
ax = sns.countplot(x=train_labels, order=order, palette=palette)
plt.xticks(rotation="vertical")
for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()), (p.get_x(), p.get_height()))
    
plt.title("Traffic distribution in the training")
plt.xlabel("Classes")
plt.ylabel("Number of samples")
# plt.savefig("distribution_up1.png", dpi=200, format='png', bbox_inches = "tight")
plt.show()

# %% [markdown]
# ### OverSampling
# 
# As we can see the distribution between classes is not balanced in the training data, so I will do some OverSampling to the minority classes using SMOTE, making the dataset balanced.

# %%
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from math import ceil
import numpy as np
import pandas as pd

# Parameters
min_threshold = 0.005  # Minimum percentage threshold for resampling
min_samples_small_class = 2  # Minimum samples required for small classes

# Ensure DataFrame/Series compatibility
def ensure_dataframe(features, labels):
    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features)
    if isinstance(labels, np.ndarray):
        labels = pd.Series(labels)
    return features.reset_index(drop=True), labels.reset_index(drop=True)

# Handle small classes separately
def replicate_small_classes(features, labels, min_samples=2):
    features, labels = ensure_dataframe(features, labels)
    small_classes = labels.value_counts()[labels.value_counts() < min_samples].index
    replicated_features, replicated_labels = [], []
    for cls in small_classes:
        cls_features = features[labels == cls]
        cls_labels = labels[labels == cls]
        replicated_features.append(resample(cls_features, replace=True, n_samples=min_samples, random_state=10))
        replicated_labels.append(resample(cls_labels, replace=True, n_samples=min_samples, random_state=10))
    if replicated_features:
        replicated_features = pd.concat(replicated_features, ignore_index=True)
        replicated_labels = pd.concat(replicated_labels, ignore_index=True)
    else:
        replicated_features = pd.DataFrame(columns=features.columns)
        replicated_labels = pd.Series(dtype=labels.dtype)
    return replicated_features, replicated_labels

# Main oversampling function
def oversample_data(features, labels):
    features, labels = ensure_dataframe(features, labels)
    labels_count = labels.value_counts()
    all_samples = labels_count.sum()

    # Handle classes with fewer than `min_samples_small_class`
    small_classes = labels_count[labels_count < min_samples_small_class].index
    small_features = features[labels.isin(small_classes)]
    small_labels = labels[labels.isin(small_classes)]
    replicated_features, replicated_labels = replicate_small_classes(small_features, small_labels, min_samples=min_samples_small_class)

    # Determine valid k_neighbors dynamically for SMOTE
    smallest_majority_class_size = labels_count[labels_count >= min_samples_small_class].min()
    k_neighbors = max(1, min(5, smallest_majority_class_size - 1))  # SMOTE requires k_neighbors < samples in class

    # Create SMOTE sampling strategy
    smote_strategy = {
        cls: max(count, ceil(min_threshold * all_samples))
        for cls, count in labels_count.items()
        if count >= min_samples_small_class
    }

    # Apply SMOTE
    smote = SMOTE(random_state=10, k_neighbors=k_neighbors, sampling_strategy=smote_strategy)
    try:
        over_features, over_labels = smote.fit_resample(features, labels)
    except ValueError as e:
        print(f"SMOTE failed with ValueError: {e}")
        return features, labels  # Return original data if SMOTE fails

    # Combine SMOTE results with small classes
    final_features = pd.concat([pd.DataFrame(over_features), replicated_features], ignore_index=True)
    final_labels = pd.concat([pd.Series(over_labels), replicated_labels], ignore_index=True)

    return final_features, final_labels

# Apply oversampling
over_train_features, over_train_labels = oversample_data(train_features, train_labels)
over_validation_features, over_validation_labels = oversample_data(validation_features, validation_labels)

# Print results
print("Oversampled Training Labels Distribution:")
print(over_train_labels.value_counts())
print("Oversampled Validation Labels Distribution:")
print(over_validation_labels.value_counts())


# %% [markdown]
# ### One-hot encoding
# 
# One-hot enconding consist in turn the categorical features into numerical features. In this dataset the only categorical feature is the "Label".
# 
# Here is applyed standardization to the features and one-hot encoding to the labels. 

# %%
from sklearn.preprocessing import OneHotEncoder

test_labels_rshped = test_labels.values.reshape(-1,1)
over_train_labels_rshped = over_train_labels.values.reshape(-1,1)
over_validation_rshped = over_validation_labels.values.reshape(-1,1)

ohenc = OneHotEncoder()


test_labels_enc = ohenc.fit_transform(test_labels_rshped).toarray()  # one-hot encoded test set lbls
over_train_labels_enc = ohenc.fit_transform(over_train_labels_rshped).toarray()  # one-hot encoded upsampled train set lbls
over_validation_labels_enc = ohenc.fit_transform(over_validation_rshped).toarray()  # one-hot encoded upsampled train set lbls for neural nets predicting upsampled traffic

print("Shape of train features", over_train_features.shape)
print("Shape of validation features", over_validation_features.shape)
print("Shape of test features", test_features.shape)

# %% [markdown]
# ## LSTM Neural Network

# %%



