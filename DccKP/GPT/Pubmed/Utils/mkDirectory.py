
import os

DIR_MODEL = "/home/javaprog/Data/Broad/GPT/Models"
ML_INTERVAL_SAVE_MODEL = 2

for i in range(10):
    if i % ML_INTERVAL_SAVE_MODEL == 0:
        # file_model = "{}/text_gen_model_state_{}.pt".format(DIR_MODEL, i)
        # torch.save(model.state_dict(), file_model)
        dir_temp = DIR_MODEL + "/text_gen_{}".format(i)
        os.mkdir(dir_temp)
        print("wrote out model for epoch: {} to file: {}".format(i, dir_temp))
