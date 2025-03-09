from run_on_dataset import run_TDC

dataset = ["50Salads", "Breakfast", "MPII_Cooking"]

dataset_name = dataset[2]

if dataset_name == "50Salads":
    [alpha, beta] = [0.11, 1.0]
elif dataset_name == "Breakfast":
    [alpha, beta] = [0.28, 2.0]
elif dataset_name == "MPII_Cooking":
    [alpha, beta] = [0.19, 1.5]

save_arr = True

Result = run_TDC(dataset_name=dataset_name,
                 datasets_path="./dataset", alpha=alpha, beta=beta, save_arr=save_arr)
