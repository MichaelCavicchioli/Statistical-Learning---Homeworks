from XGB_Parameter_Tuning_class import *

if __name__ == "__main__":
    tuner = XGB_Parameter_Tuning()
    tuner.preprocess_data()
    tuner.split_data()
    tuner.run_tuning()
    tuner.train_best_model()
    tuner.evaluate_model()
    tuner.grid_search()