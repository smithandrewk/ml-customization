OOps. I had this code during the most recent experiment

from lib.train_utils import random_subsample
from lib.train_utils import append_losses_and_f1

if participants:
    print(f'Base train dataset size: {len(base_train_dataset)}')
    base_train_dataset = random_subsample(base_train_dataset, 1)
    print(f'Base train dataset size: {len(base_train_dataset)}')

    print(f'Base val dataset size: {len(base_val_dataset)}')
    base_val_dataset = random_subsample(base_val_dataset, .5)
    print(f'Base val dataset size: {len(base_val_dataset)}')

    base_trainloader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
    base_valloader = DataLoader(base_val_dataset, batch_size=batch_size, shuffle=False)

# Subsample target training data if specified
target_data_pct = hyperparameters['target_data_pct']
if target_data_pct < 1.0:
    print(f'Target train dataset size: {len(target_train_dataset)}')
    target_train_dataset = random_subsample(target_train_dataset, target_data_pct)
    print(f'Target train dataset size: {len(target_train_dataset)}')

print(f'Target val dataset size: {len(target_val_dataset)}')
target_val_dataset = random_subsample(target_val_dataset, .1)
print(f'Target val dataset size: {len(target_val_dataset)}')

But things worked..

Leave-one-participant-out mode: using alsaad as target participant.                                                                                                                                                  09:05:01 [180/1815]
Using 6 base participants: ['tonmoy', 'asfik', 'anam', 'ejaz', 'iftakhar', 'unk1']                                                                                                                                                      
Using test model: TestModel                                                                                                                                                                                                             
Total model parameters: 969                                                                                                                                                                                                             
Base train dataset size: 77498                                                                                                                                                                                                          
Subsampling dataset to 100% of original size                                                                                                                                                                                            
Base train dataset size: 77498                                                                                                                                                                                                          
Base val dataset size: 31060                                                                                                                                                                                                            
Subsampling dataset to 50.0% of original size                                                                                                                                                                                           
Base val dataset size: 15530                                                                                                                                                                                                            
Target train dataset size: 11871                                                                                                                                                                                                        
Subsampling dataset to 50.0% of original size                                                                                                                                                                                           
Target train dataset size: 5935                                                                                                                                                                                                         
Target val dataset size: 4141                                                                                                                                                                                                           
Subsampling dataset to 10.0% of original size                                                                                                                                                                                           
Target val dataset size: 414                                                                                                                                                                                                            
Target train dataset size: 5935                                                                                                                                                                                                         
Target val dataset size: 414 