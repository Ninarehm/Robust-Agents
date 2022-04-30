# Robust Conversational Agents against Imperceptible Toxicity Triggers

# To run the Attacks:

```bash
cd Attacks
```
##For UTC attacks run:
```bash
python UTC_runner.py --device_type cuda --dataset reddit --classifier perspective-safety --criteria 2
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset, classifier can be replace by toxicity classifiers discussed in the paper, criteria 1,2, or 3.

##For UTA attack run:
```bash
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

##For UTA-LM attack run:
```bash
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA-LM
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

##For no attack run:
```bash
python vanilla_runner.py --device_type cuda --dataset reddit
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

# To run the Defenses:

```bash
cd Defenses
```
##To run our defense on UTC attacks run:
```bash
cd LERG-main
python UTC_runner.py --device_type cuda --dataset reddit --classifier perspective-safety --criteria 2
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset, classifier can be replace by toxicity classifiers discussed in the paper, criteria 1,2, or 3.

##To run our defense on UTA attack run:
```bash
cd LERG-main
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

##To run our defense on UTA-LM attack run:
```bash
cd LERG-main
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA-LM
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.


##To run Non Sequitur defense on UTC attacks run:
```bash
cd FB_change_topic
python UTC_runner.py --device_type cuda --dataset reddit --classifier perspective-safety --criteria 2
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset, classifier can be replace by toxicity classifiers discussed in the paper, criteria 1,2, or 3.

##To run Non Sequitur defense on UTA attack run:
```bash
cd FB_change_topic
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

##To run Non Sequitur defense on UTA-LM attack run:
```bash
cd FB_change_topic
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA-LM
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.


##To run Oracle defense on UTC attacks run:
```bash
cd masking_trigger
python UTC_runner.py --device_type cuda --dataset reddit --classifier perspective-safety --criteria 2
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset, classifier can be replace by toxicity classifiers discussed in the paper, criteria 1,2, or 3.

##To run Oracle defense on UTA attack run:
```bash
cd masking_trigger
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

##To run Oracle defense on UTA-LM attack run:
```bash
cd masking_trigger
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA-LM
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

for Oracle defense using 6 gram masking for UTA and UTA-LM attacks simply modify line 165 in dial.py to attn_mask[:,adv_starting_idx:] = 0