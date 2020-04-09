import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("prediction_dir", help="The directory that contains the folders with the model names(must end with /)")
parser.add_argument("configuration_file", help="The configuration file where each line starts with the model name, continues with the classes which are prefered to be taken from the respective model, separated with space character(must end with /)")

parser.add_argument("output_dir", help="The directory which will contain the outputs(must end with /)")

args = parser.parse_args()

prediction_dir = args.prediction_dir
config_file = args.configuration_file
output_dir = args.output_dir

model_names = []

classes = [  'specularity',
             'saturation',
             'artifact',
             'blur',
             'contrast',
             'bubbles',
             'instrument',
             'blood'
        ]


def create_pref_dict():
    pref_dict = {}

    with open(config_file, "r") as f:
        content = f.read()
        lines = content.split("\n")

        for line in lines:
            line_content = line.split(" ")
            model = line_content[0]
            if model in model_names:
                pref_dict[model] = line_content[1:]
            else:
                print("Model name {} was define inside the configuration file {}, but could not be found in the folder {}".format(model, config_file, prediction_dir))
                exit(-1)

    return pref_dict



def main():

    for model in glob.glob(prediction_dir + "*\\"):
        name = model.split('\\')[-2].split('.')[0]
        model_names.append(name)

    pref_dict = create_pref_dict()
    print(str(pref_dict))

    new_file_content = {}

    for model in model_names:
        cur_dir = prediction_dir + model
        print(cur_dir)
        for cur_file_name in glob.glob(cur_dir + "\\*.txt"):
            new_file_name = cur_file_name.split("\\")[-1]
            if new_file_name not in new_file_content.keys():
                new_file_content[new_file_name] = ""
            with open(cur_file_name, "r") as f:
                content = f.read()
                lines = content.split("\n")
                for line in lines:
                    class_name = line.split(" ")[0]
                    if class_name in pref_dict[model]:
                        new_file_content[new_file_name] += line + "\n"

    for file_name in new_file_content.keys():
        write_file = output_dir + file_name
        with open(write_file, "w") as f:
            f.write(new_file_content[file_name])

if __name__ == '__main__':
    main()
