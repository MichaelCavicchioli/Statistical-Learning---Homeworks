from utility import load_ds, clear_df, knn, cross_validation

# Carica il dataset
df = load_ds()

# Pulizia del dataframe
df = clear_df(df)

### KNN ###
knn, X, y = knn(df)
###########

### CV ###
cross_validation(knn, X, y)
###########