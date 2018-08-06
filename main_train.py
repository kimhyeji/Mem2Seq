import numpy as np
import logging 
from tqdm import tqdm

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.Mem2Seq import *

import nsml

if nsml.IS_ON_NSML:
    from nsml_helper import bind_model

BLEU = False

if (args['decoder'] == "Mem2Seq"):
    if args['dataset']=='kvr':
        from utils.utils_kvr_mem2seq import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi_mem2seq import *
    else: 
        print("You need to provide the --dataset information")
else:
    if args['dataset']=='kvr':
        from utils.utils_kvr import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi import *
    else: 
        print("You need to provide the --dataset information")

# Configure models
avg_best,cnt,acc = 0.0,0,0.0
cnt_1 = 0
### LOAD DATA
train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'],batch_size=int(args['batch']),shuffle=True)

if args['decoder'] == "Mem2Seq":
    model = globals()[args['decoder']](int(args['hidden']),
                                        max_len,max_r,lang,args['path'],args['task'],
                                        lr=float(args['learn']),
                                        n_layers=int(args['layer']), 
                                        dropout=float(args['drop']),
                                        unk_mask=bool(int(args['unk_mask']))
                                    )
else:
    model = globals()[args['decoder']](int(args['hidden']),
                                    max_len,max_r,lang,args['path'],args['task'],
                                    lr=float(args['learn']),
                                    n_layers=int(args['layer']), 
                                    dropout=float(args['drop'])
                                )

if nsml.IS_ON_NSML:
    bind_model(model)

for epoch in range(100):
    logging.info("Epoch:{}".format(epoch))  
    # Run the train function
    if nsml.IS_ON_NSML:
        pbar = enumerate(train)
    else:
        pbar = tqdm(enumerate(train), total=len(train))

    for i, data in pbar:
        if args['decoder'] == "Mem2Seq":
            if args['dataset']=='kvr':
                model.train_batch(data[0], data[1], data[2], data[3],data[4],data[5],
                            len(data[1]),10.0,0.5, data[-2], data[-1],i==0)
            else:
                model.train_batch(data[0], data[1], data[2], data[3],data[4],data[5],
                            len(data[1]),10.0,0.5, data[-4], data[-3],i==0)
        else:
            model.train_batch(data[0], data[1], data[2], data[3],data[4],data[5],
                        len(data[1]),10.0,0.5,i==0)
    
    if((epoch+1) % int(args['evalp']) == 0):    
        acc = model.evaluate(dev,avg_best, epoch, BLEU)
        acc_test = model.evaluate(test, 1e6, BLEU)

        if args['task'] != 6:
            acc_oov_test = model.evaluate(testOOV, 1e6, BLEU)
            print("ACC-DEV,TEST,OOV {} {} {}".format(str(acc),str(acc_test),str(acc_oov_test)))
        else:
            print("ACC-DEV,TEST {} {}".format(str(acc), str(acc_test)))

        if 'Mem2Seq' in args['decoder']:
            model.scheduler.step(acc)
        if(acc >= avg_best):
            avg_best = acc
            cnt=0
        else:
            cnt+=1
        #if(acc_test == 1.0): break
        if nsml.IS_ON_NSML:
            nsml.save(epoch)




