# Split train document into many documents of different sizes

# Copy test and label files to that folder

# Run NER on each folder sequentially, rename Wandb run appropriately

# Compute metrics on the output

# Plot all output metrics

"""Run training curve over different dataset sizes for NER"""

# imports
from os import path, mkdir, environ, system
from collections import namedtuple
import shutil
# import wandb
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    help="splitter, trainer or plotter",
    type=str
)

class SplitTrain:
    def __init__(self):
        self.data_folder = '/data/pranav/projects/polymer_ner/data/polymer_dataset_labeling_7/'
        self.train_file = path.join(self.data_folder, 'train.json')
        self.train_split_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.output_dir_template = path.join(self.data_folder, 'train_split_')
        self.total_docs = 652
    
    def split_train_file(self):
        # parse each line
        # after each document check if num docs is equal to that in train split ratios, if so write to a separate file
        # token_labels = []
        token_label = namedtuple('token_label', ["text", "label"])
        token_label_list = []
        num_docs = 0
        # with open(self.train_file, 'r') as fi:
        #     for line in fi:
        #         if line != '\n':
        #             rec = line.split()
        #             token_labels.append(token_label(rec[0], rec[1]))
        #         else:
        #             num_docs+=1
        #             token_label_list.append(token_labels)
        #             for ratio in self.train_split_ratios:
        #                 if num_docs == int(self.total_docs*ratio):
        #                     output_dir = self.output_dir_template+str(ratio)
        #                     if not path.exists(output_dir):
        #                         mkdir(output_dir)
        #                     self.write_output_file(token_label_list, path.join(output_dir, 'train.txt'))
        #                     break
        #             token_labels = []
        with open(self.train_file) as fi:
            for num_docs, line in enumerate(fi):
                line_info = json.loads(line)
                token_label_list.append([token_label(word, label) for word, label in zip(line_info['words'], line_info['ner'])])
                for ratio in self.train_split_ratios:
                    if num_docs == int(self.total_docs*ratio)-1:
                        output_dir = self.output_dir_template+str(ratio)
                        if not path.exists(output_dir):
                            mkdir(output_dir)
                        self.write_output_file(token_label_list, path.join(output_dir, 'train.json'))
                        break

    def prep_folder(self):
        # Copy remaining necessary files over to the output dirs
        for ratio in self.train_split_ratios:
            # copy dev test and label file
            output_dir = self.output_dir_template+str(ratio)
            for file in ['dev.json', 'test.json', 'labels.txt', 'dev.txt', 'test.txt']:
                shutil.copy(path.join(self.data_folder, file), output_dir)
    
    # Code taken from dataset_conversion.py in polymer_ner - might have been possible to inherit from it. Consider this further
    def write_output_file(self, token_list, output_file):
        with open(output_file, 'w') as fo:
            for tokens in token_list:
                line_dict = {"words": [], "ner": []}
                for token in tokens:
                    line_dict["words"].append(token.text)
                    line_dict["ner"].append(token.label)
                fo.write(json.dumps(line_dict))
                fo.write('\n')
    
    def run(self):
        self.split_train_file()
        self.prep_folder()


class TrainCurveNER(SplitTrain):
    def __init__(self):
        super(TrainCurveNER, self).__init__()
        # wandb.init(project="polymer_ner")
        # os.environ['WANDB_DIR'] = '/data/pranav/projects/polymer_ner/data/'

    def run_ner_train_curve(self):
        for ratio in self.train_split_ratios:
            output_dir = self.output_dir_template+str(ratio)
            if not path.exists(path.join(output_dir, 'log')): mkdir(path.join(output_dir, 'log'))
            if not path.exists(path.join(output_dir, 'output')): mkdir(path.join(output_dir, 'output'))
            # Set wandb_output_name
            exp_name = f'PubMedBertFull_polymer_dataset_labeling_7_7_epochs_train_split_{ratio}'
            environ['WANDB_NAME'] = exp_name
            # Not certain if environment will seep down to the process
            system(f'python3 -u /home/pranav/repos/transformers-4.17.0/examples/pytorch/token-classification/run_ner.py \
                     --model_name_or_path /data/pranav/projects/matbert/pretrained_models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/ \
                     --train_file {path.join(output_dir, "train.json")}\
                     --validation_file {path.join(output_dir, "dev.json")}\
                     --test_file {path.join(output_dir, "test.json")} \
                     --exp_name {exp_name} \
                     --output_dir {output_dir}/output/PubMedBertFull \
                     --evaluation_strategy="epoch" \
                     --overwrite_output_dir \
                     --max_seq_len 512 \
                     --num_train_epochs 7 \
                     --logging_steps 2 \
                     --do_train \
                     --do_eval \
                     --do_predict \
                     --overwrite_cache > {output_dir}/log/log_ner_train_PubMedBertFull_ratio_{ratio}')


class ComputeMetrics(SplitTrain):
    # Compute metrics automatically and plot them
    def __init__(self):
        super(ComputeMetrics, self).__init__()
        self.entities = ['macro', 'POLYMER', 'POLYMER_FAMILY', 'MONOMER', 'ORGANIC', 'INORGANIC', 'PROP_NAME', 'PROP_VALUE', 'MATERIAL_AMOUNT']
        self.compute_metrics_file = '/home/pranav/repos/polymer_ner/polymer_ner/compute_metrics.py'
        self.plot_dir = '/data/pranav/projects/polymer_ner/figures'
    
    def compute_metrics(self):
        for ratio in self.train_split_ratios:
            output_dir = self.output_dir_template+str(ratio)
            system(f'python -u {self.compute_metrics_file} --ground_truth_file {output_dir}/test.txt --predictions_file {output_dir}/output/PubMedBertFull/test_predictions.txt --label_file {output_dir}/labels.txt --output_file {output_dir}/output/PubMedBertFull/detailed_metrics.json')
    
    def retrieve_metrics(self):
        metrics_dict = {}
        for ratio in self.train_split_ratios:
            metrics_file = self.output_dir_template+str(ratio)+'/output/PubMedBertFull/detailed_metrics.json'
            metrics_dict[ratio]={}
            with open(metrics_file, 'r') as fi:
                ner_metrics = json.load(fi)
            for entity in self.entities:
                metrics_dict[ratio][entity]={}
                metrics_dict[ratio][entity]['entity_precision'] = ner_metrics[entity]['entity_precision']
                metrics_dict[ratio][entity]['entity_recall'] = ner_metrics[entity]['entity_recall']
                metrics_dict[ratio][entity]['entity_f1'] = ner_metrics[entity]['entity_f1']
        
        return metrics_dict

    def plot_overall(self, metrics_dict):
        # Make one plot of overall P, R, F1 across dataset sizes
        precision = [metrics_dict[ratio]['macro']['entity_precision'] for ratio in self.train_split_ratios]
        recall = [metrics_dict[ratio]['macro']['entity_recall'] for ratio in self.train_split_ratios]
        f1 = [metrics_dict[ratio]['macro']['entity_f1'] for ratio in self.train_split_ratios]
        fig, ax = plt.subplots()

        ax.plot(self.train_split_ratios, precision, label='precision')
        ax.plot(self.train_split_ratios, recall, label='recall')
        ax.plot(self.train_split_ratios, f1, label='f1')
        ax.scatter(self.train_split_ratios, precision)
        ax.scatter(self.train_split_ratios, recall)
        ax.scatter(self.train_split_ratios, f1)
        ax.set_xlabel('Training data split ratio')
        ax.set_ylabel('Performance Metrics')
        ax.set_ylim(0,1)
        ax.set_title('Overall performance')
        ax.legend()
        fig.savefig(path.join(self.plot_dir, 'overall_fig_PubMedBertFull.png'))


    def single_plot(self, ax, label):
        ax.set_xlabel('Train Split ratio')
        ax.set_ylabel(f'{label} score')
        ax.set_ylim(0,1)
        # ax.legend()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    
    def plot_entities(self, metrics_dict):
        # Make a plot each of P, R, F1 for each entity type using entity F1 scores
        fig_p, ax_p = plt.subplots()
        fig_r, ax_r = plt.subplots()
        fig_f, ax_f = plt.subplots()

        for entity in self.entities:
            precision_list = [metrics_dict[ratio][entity]['entity_precision'] for ratio in self.train_split_ratios]
            recall_list = [metrics_dict[ratio][entity]['entity_recall'] for ratio in self.train_split_ratios]
            f1_list = [metrics_dict[ratio][entity]['entity_f1'] for ratio in self.train_split_ratios]
            if entity=='macro':
                entity='overall'
            elif entity=='PROP_NAME':
                entity = 'PROPERTY_NAME'
            elif entity == 'PROP_VALUE':
                entity = 'PROPERTY_VALUE'

            ax_p.plot(self.train_split_ratios, precision_list, label=entity)
            ax_r.plot(self.train_split_ratios, recall_list, label=entity)
            ax_f.plot(self.train_split_ratios, f1_list, label=entity, linewidth=1.5)
            ax_p.scatter(self.train_split_ratios, precision_list,  marker='x', s=24)
            ax_r.scatter(self.train_split_ratios, recall_list,  marker='x', s=24)
            ax_f.scatter(self.train_split_ratios, f1_list, marker='x', s=28)

        self.single_plot(ax_r, 'Recall')
        self.single_plot(ax_p, 'Precision')
        self.single_plot(ax_f, 'F1')
        fig_r.savefig(path.join(self.plot_dir, 'recall_entities_PubMedBertFull.png'))
        fig_p.savefig(path.join(self.plot_dir, 'precision_entities_PubMedBertFull.png'))
        fig_f.savefig(path.join(self.plot_dir, 'f1_entities_PubMedBertFull.png'))

    def run(self):
        self.compute_metrics()
        metrics_dict = self.retrieve_metrics()
        self.plot_overall(metrics_dict)
        self.plot_entities(metrics_dict)



if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode=='splitter':
        splitter = SplitTrain()
        splitter.run()
    if args.mode=='trainer':
        trainer = TrainCurveNER()
        trainer.run_ner_train_curve()
    elif args.mode=='plotter':
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.use('Cairo')
        from automation_utilities import plot_config
        plot_config.config_publications(mpl)
        plotter = ComputeMetrics()
        plotter.run()
