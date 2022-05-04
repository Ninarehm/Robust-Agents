# Robust Conversational Agents against Imperceptible Toxicity Triggers

# To run the Attacks:

```bash
cd Attacks
```
**For UTSC attacks run:**
```bash
python UTC_runner.py --device_type cuda --dataset reddit --classifier perspective-safety --criteria 2
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset, classifier can be replace by toxicity classifiers discussed in the paper, criteria 1,2, or 3.

**For UAT attack run:**
```bash
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

**For UAT-LM attack run:**
```bash
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA-LM
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

**For no attack run:**
```bash
python vanilla_runner.py --device_type cuda --dataset reddit
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

# To run the Defenses:

```bash
cd Defenses
```
**To run our defense on UTCS attacks run:**
```bash
cd LERG-main
python UTC_runner.py --device_type cuda --dataset reddit --classifier perspective-safety --criteria 2
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset, classifier can be replace by toxicity classifiers discussed in the paper, criteria 1,2, or 3.

**To run our defense on UAT attack run:**
```bash
cd LERG-main
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

**To run our defense on UAT-LM attack run:**
```bash
cd LERG-main
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA-LM
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.


**To run Non Sequitur defense on UTCS attacks run:**
```bash
cd FB_change_topic
python UTC_runner.py --device_type cuda --dataset reddit --classifier perspective-safety --criteria 2
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset, classifier can be replace by toxicity classifiers discussed in the paper, criteria 1,2, or 3.

**To run Non Sequitur defense on UAT attack run:**
```bash
cd FB_change_topic
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

**To run Non Sequitur defense on UAT-LM attack run:**
```bash
cd FB_change_topic
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA-LM
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.


**To run Oracle defense on UTCS attacks run:**
```bash
cd masking_trigger
python UTC_runner.py --device_type cuda --dataset reddit --classifier perspective-safety --criteria 2
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset, classifier can be replace by toxicity classifiers discussed in the paper, criteria 1,2, or 3.

**To run Oracle defense on UAT attack run:**
```bash
cd masking_trigger
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

**To run Oracle defense on UAT-LM attack run:**
```bash
cd masking_trigger
python UTA_UTA-LM_runner.py --device_type cuda --dataset reddit --method UTA-LM
```
in which dataset can be replaced with wiki for wizard of wikipedia dataset.

# Notes 

Note1: for Oracle defense using 6 gram masking for UAT and UAT-LM attacks simply modify line 165 in dial.py to attn_mask[:,adv_starting_idx:] = 0


Note2: To run the experiments, you will need Perspective API's key credientials. Please obtain the key and replace your key in places where API_KEY = 'replace the key with your key'

Note3: For questions please reach out to ninarehm at usc dot edu