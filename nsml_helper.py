import torch
import nsml
from utils.config import args as config
if (config['decoder'] == "Mem2Seq"):
    if config['dataset'] == 'kvr':
        from utils.utils_kvr_mem2seq import *

        BLEU = True
    elif config['dataset'] == 'babi':
        from utils.utils_babi_mem2seq import *
    else:
        print("You need to provide the --dataset information")
else:
    if config['dataset'] == 'kvr':
        from utils.utils_kvr import *

        BLEU = True
    elif config['dataset'] == 'babi':
        from utils.utils_babi import *
    else:
        print("You need to provide the --dataset information")

def bind_model(model, **kwargs):
    def save(filename):
        # save the model with 'checkpoint' dictionary.
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)
        print("Model saved")

    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        BLEU = False
        acc_oov_test = 0

        train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                           shuffle=True)
        acc_test = model.evaluate(test, 1e6, BLEU)
        print(acc_test)
        if testOOV != []:
            acc_oov_test = model.evaluate(testOOV, 1e6, BLEU)
            print(acc_oov_test)
        return (acc_test, acc_oov_test)

    # function in function is just used to divide the namespace.
    nsml.bind(save=save, load=load, infer=infer)