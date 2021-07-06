from utils.general import get_logger_old
import os
from os import path
import argparse




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="The target device.")
    parser.add_argument('-t','--test',required=True, type=str, choices=['fvd','accuracy','diversity', 'kps_acc'],help="Which test to conduct.")
    args = parser.parse_args()



    with open("config/model_names.txt", "r") as f:
        model_names = f.readlines()
    model_names = [m for m in model_names if not m.startswith("#")]
    file = path.basename(__file__)
    logger = get_logger_old(file)


    gpu = args.gpu

    for n in model_names:
        n = n.rstrip()
        logger.info(f'Conducting experiment "{args.test}" for model {n}')

        try:
            test_cmd = f"python -W ignore main.py --config config/second_stage.yaml --gpus {gpu} --model_name {n} --test {args.test}"
            if args.test == 'fvd' and "LD_LIBRARY_PATH" in os.environ:
                test_cmd = f'LD_LIBRARY_PATH={os.environ["LD_LIBRARY_PATH"]} ' + test_cmd
            os.system(test_cmd)
        except Exception as e:
            logger.error(e)
            logger.info("next model")
            continue

    logger.info("finished")