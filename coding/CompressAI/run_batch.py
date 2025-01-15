import os
import re
import matplotlib.pyplot as plt
import time

os.environ['MKL_THREADING_LAYER'] = 'GNU'


def generate_commands(data_root, model_name, model_type, task, max_v, min_v, trun_flag, trun_high, trun_low, quant_type, samples, bit_depth, arch, lambda_value, epochs, learning_rate, batch_size, patch_size):
    if isinstance(trun_low, list):
        trun_low = '[' + ','.join(map(str, trun_low)) + ']'
        trun_high = '[' + ','.join(map(str, trun_high)) + ']'

    # generate train command
    train_command = (
        f"python examples/train.py --model {arch} "
        f"-d {data_root}/{model_type}/{task}/feature_train "
        f"--lambda {lambda_value} --epochs {epochs} -lr {learning_rate} "
        f"--batch-size {batch_size} --patch-size {patch_size} --cuda --save "
        f"--model_type={model_type} --task={task} --trun_flag={trun_flag} --trun_low={trun_low} --trun_high={trun_high} --quant_type={quant_type} --qsamples={samples} --bit_depth={bit_depth} "
        f"-mp {data_root}/{model_type}/{task}/{model_name}/training_models/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"{model_name}_lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}_checkpoint.pth.tar "
        f">{data_root}/{model_type}/{task}/{model_name}/training_log/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"train_{model_name}_lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}.txt 2>&1"
    )

    # generate eval command
    eval_command = (
        f"python -m compressai.utils.eval_model checkpoint "
        f"{data_root}/{model_type}/{task}/feature_test "
        f"-a {arch} --cuda -v "
        f"-d {data_root}/{model_type}/{task}/{model_name}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')} "
        f"--per-image -p {data_root}/{model_type}/{task}/{model_name}/training_models/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"{model_name}_lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}_checkpoint.pth.tar "
        f"--model_type={model_type} --task={task} --trun_flag={trun_flag} --trun_low={trun_low} --trun_high={trun_high} --quant_type={quant_type} --qsamples={samples} --bit_depth={bit_depth} "
        f">{data_root}/{model_type}/{task}/{model_name}/encoding_log/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"compress_{model_name}_lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}.txt 2>&1"
    )

    return train_command, eval_command

def plot_train_loss(train_log_path, train_config):
    # Read all content
    log_name = os.path.join(train_log_path, train_config+'.txt')
    pdf_name = os.path.join(train_log_path, train_config+'.pdf')
    with open(log_name, 'r') as file:
        file_content = file.read()

    # Extract content that starts with "Test epoch"
    test_epoch_lines = re.findall(r"Test epoch.*", file_content)

    # Init
    losses = []
    mse_losses = []
    bpp_losses = []
    aux_losses = []

    # Extract losses
    for line in test_epoch_lines:
        match = re.search(r"Loss:\s*([\d\.]+)\s*\|\s*MSE\s*loss:\s*([\d\.]+)\s*\|\s*Bpp\s*loss:\s*([\d\.]+)\s*\|\s*Aux\s*loss:\s*([\d\.]+)", line)
        if match:
            losses.append(float(match.group(1)))
            mse_losses.append(float(match.group(2)))
            bpp_losses.append(float(match.group(3)))
            aux_losses.append(float(match.group(4)))

    epochs = range(len(losses))

    plt.figure(figsize=(12,8))

    plt.subplot(221)
    plt.plot(epochs, losses, marker='o')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(222)
    plt.plot(epochs, mse_losses, marker='o')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')

    plt.subplot(223)
    plt.plot(epochs, bpp_losses, marker='o')
    plt.title('Bpp Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Bpp Loss')

    plt.subplot(224)
    plt.plot(epochs, aux_losses, marker='o')
    plt.title('Aux Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Aux Loss')

    plt.tight_layout()
    plt.savefig(pdf_name, dpi=600, format='pdf', bbox_inches='tight')

def hyperprior_train_evaluate_pipeline(data_root, model_type, task, max_v, min_v, trun_flag, trun_low, trun_high, quant_type, samples, bit_depth, lambda_value, epochs, learning_rate, batch_size, patch_size):
    arch = 'bmshj2018-hyperprior'
    model_name = arch.split('-')[-1]; print(model_name)

    if trun_flag == False: trun_high = max_v; trun_low = min_v

    quant_type = 'uniform'; samples = 0; bit_depth = 1

    train_cmd, eval_cmd = generate_commands(data_root, model_name, model_type, task, max_v, min_v, trun_flag, trun_high, trun_low, quant_type, samples, bit_depth, arch, lambda_value, epochs, learning_rate, batch_size, patch_size)

    # time_start = time.time()
    # print("Train Command:")
    # print(train_cmd)
    # os.system(train_cmd)
    # print('training time: ', time.time() - time_start)

    # train_log_path = f"{data_root}/{model_type}/{task}/{model_name}/training_log/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
    # train_config = f"train_{model_name}_lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"
    # plot_train_loss(train_log_path, train_config)

    time_start = time.time()
    print("\nEval Command:")
    print(eval_cmd)
    os.system(eval_cmd)
    print('encoding time: ', time.time() - time_start)

if __name__ == "__main__":
    model_type = 'llama3'; task = 'csr'
    max_v = 47.75; min_v = -78; trun_high = 5; trun_low = -5
    lambda_value_all = [0.01405, 0.0142, 0.015, 0.07, 10]
    epochs = 200; learning_rate = "1e-4"; batch_size = 40; patch_size = "64 4096" # height first, width later

    # model_type = 'dinov2'; task = 'cls'
    # max_v = 104.1752; min_v = -552.451; trun_high = 5; trun_low = -5
    # lambda_value_all = [0.001, 0.0017, 0.003, 0.0035, 0.01]
    # epochs = 800; learning_rate = "1e-4"; batch_size = 128; patch_size = "256 256"   # height first, width later

    # model_type = 'dinov2'; task = 'seg'
    # max_v = 103.2168; min_v = -530.9767; trun_high = 5; trun_low = -5
    # lambda_value_all = [0.0005, 0.001, 0.003, 0.007, 0.015]
    # epochs = 800; learning_rate = "1e-4"; batch_size = 128; patch_size = "256 256"   # height first, width later

    # model_type = 'dinov2'; task = 'dpt'
    # max_v = [3.2777, 5.0291, 25.0456, 102.0307]; min_v = [-2.4246, -26.8908, -323.2952, -504.4310]; trun_high = [1,2,10,10]; trun_low = [-1,-2,-10,-10]
    # lambda_value_all = [0.001, 0.005, 0.02, 0.05, 0.12]
    # epochs = 200; learning_rate = "1e-4"; batch_size = 128; patch_size = "256 256"   # height first, width later
    
    # model_type = 'sd3'; task = 'tti'
    # max_v = 4.668; min_v = -6.176; trun_high = 4.668; trun_low = -6.176
    # lambda_value_all = [0.005, 0.01, 0.02, 0.05, 0.2]
    # epochs = 60; learning_rate = "1e-4"; batch_size = 32; patch_size = "512 512"   # height first, width later

    trun_flag = True
    quant_type = 'uniform'; samples = 0; bit_depth = 1
    data_root = "/home/gaocs/projects/FCM-LM/Data"

    # lambda_value_all = [0.01, 0.02, 0.05, 0.2]
    for lambda_value in lambda_value_all:
        hyperprior_train_evaluate_pipeline(data_root, model_type, task, max_v, min_v, trun_flag, trun_low, trun_high, quant_type, samples, bit_depth, lambda_value, epochs, learning_rate, batch_size, patch_size)