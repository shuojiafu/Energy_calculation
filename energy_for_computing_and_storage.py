#
import pandas as pd
import numpy as np


#%% Calculate Data Storage Requirement
def data_storage_split(D_daily, R_days, R_factors, O_index,
                       B_snapshot, B_freq, hot_data_ratio=0.2):
    """
    Determine total storage requirement and split into SSD and HDD.

    Parameters:
    - D_daily: Daily data generated (TB)
    - R_days: Retention period (days)
    - R_factors: Replication factor
    - O_index: Overhead (e.g., 0.2 for 20%)
    - B_snapshot: Full backup size (TB)
    - B_freq: Full backup frequency per year
    - hot_data_ratio: Fraction of data that must be on SSD (0 to 1)

    Returns:
    - total_storage (TB), ssd_storage (TB), hdd_storage (TB)
    """
    raw_storage = D_daily * R_days
    metadata_overload = raw_storage * O_index
    backup = B_snapshot * B_freq
    total_storage = (raw_storage + metadata_overload) * R_factors + backup

    ssd_storage = total_storage * hot_data_ratio
    hdd_storage = total_storage - ssd_storage

    return total_storage, ssd_storage, hdd_storage
#%%
import math

def estimate_storage_servers_energy(ssd_storage_tb,
                                        hdd_storage_tb,duration_hours=24,
                                        ssd_capacity_per_server_tb=20,
                                        hdd_capacity_per_server_tb=100,
                                        ssd_power_per_server_watts=300,
                                        hdd_power_per_server_watts=400):
    """
    Estimate number of SSD and HDD servers and their hourly energy usage.

    Args:
        ssd_storage_tb: Total SSD storage needed (TB)
        hdd_storage_tb: Total HDD storage needed (TB)
        ssd_capacity_per_server_tb: Capacity of one SSD server (TB)
        hdd_capacity_per_server_tb: Capacity of one HDD server (TB)
        ssd_power_per_server_watts: Power draw per SSD server (W)
        hdd_power_per_server_watts: Power draw per HDD server (W)

    Returns:
        Dictionary with counts and hourly energy in kWh
    """
    num_ssd_servers = math.ceil(ssd_storage_tb / ssd_capacity_per_server_tb)
    num_hdd_servers = math.ceil(hdd_storage_tb / hdd_capacity_per_server_tb)

    ssd_energy_kwh_per_day = (num_ssd_servers * ssd_power_per_server_watts)*duration_hours / 1000
    hdd_energy_kwh_per_day = (num_hdd_servers * hdd_power_per_server_watts) *duration_hours/ 1000

    return {
        "num_ssd_servers": num_ssd_servers,
        "num_hdd_servers": num_hdd_servers,
        "ssd_energy_kwh_per_day": ssd_energy_kwh_per_day,
        "hdd_energy_kwh_per_day": hdd_energy_kwh_per_day,
        "total_energy_kwh_per_day": ssd_energy_kwh_per_day + hdd_energy_kwh_per_day
    }


#%%
def estimate_compute_servers_energy(cpu_per_job, ram_per_job, gpu_per_job,
                                 jobs_per_day, hours_per_job, gpu_flops_per_sec,cpu_flops_per_sec, inferencechoice,
                                 inference_qps, inference_flops, duration_hours=24,
                                 cpu_power_watt=85, gpu_power_watt=350, ram_power_watt=3,
                                 cost_per_kwh=0.12):
    """
    Estimate compute, energy use, and cost at the data center level for both training and inference workloads.

    Parameters:
    - cpu_per_job: Number of CPU cores required per training job
    - ram_per_job: Amount of RAM (in GB) required per training job
    - gpu_per_job: Number of GPUs required per training job
    - jobs_per_day: Total number of training jobs processed per day
    - hours_per_job: Duration of each training job in hours
    - gpu_flops_per_sec: Performance of a single GPU in FLOPs per second (e.g., A100 ≈ 312e12)
    - cpu_flops_per_sec: Performance of a single CPU in FLOPs per second (e.g., A100 ≈ 312e12)
    - inference_qps: Average number of inference queries per second (QPS)
    - inference_flops: Number of FLOPs required to process a single inference query
    - duration_hours: Duration of the estimation window (default is 24 hours = 1 day)
    - cpu_power_watt: Power consumption per CPU core in watts (default: 85 W)
    - gpu_power_watt: Power consumption per GPU in watts (default: 350 W)
    - ram_power_watt: Power consumption per GB of RAM in watts (default: 3 W)
    - cost_per_kwh: Electricity cost per kilowatt-hour in USD (default: $0.12)

    Returns:
    A dictionary containing:
    - Resource provisioning (CPU cores, RAM, GPU hours)
    - Compute estimates (training and inference FLOPs per day)
    - Energy usage for training and inference
    - Total energy (in Wh and kWh)
    - Estimated daily energy cost (USD)
    """

    # Step 1: Estimate average concurrent training jobs
    concurrent_jobs = jobs_per_day * hours_per_job / duration_hours

    # Step 2: Estimate average concurrent CPU and RAM usage
    total_cpu = cpu_per_job * concurrent_jobs
    total_ram = ram_per_job * concurrent_jobs

    # Step 3: Total GPU usage (training) across all jobs per day
    total_gpu_hours_training = gpu_per_job * hours_per_job * jobs_per_day

    # Step 4: CPU and RAM usage in hours (based on concurrency)
    total_cpu_hours = total_cpu * duration_hours
    total_ram_hours = total_ram * duration_hours

    # Step 5: Compute total training FLOPs (GPU-hours × FLOPs/sec × 3600 sec/hr)
    training_flops = total_gpu_hours_training * gpu_flops_per_sec * 3600

    # Step 6: Compute total inference FLOPs
    inference_flops_total = inference_qps * inference_flops * duration_hours * 3600

    # Step 7: Estimate inference GPU usage (GPU-hours)
    # = Total inference FLOPs / (FLOPs/sec × seconds/hour)
    if inferencechoice=='gpu':
      inference_gpu_hours = inference_flops_total / (gpu_flops_per_sec * 3600)
    else:
      inference_gpu_hours = inference_flops_total / (cpu_flops_per_sec * 3600)

    # Step 8: Energy estimates (Wh)
    energy_cpu = total_cpu_hours * cpu_power_watt
    energy_ram = total_ram_hours * ram_power_watt
    energy_gpu_training = total_gpu_hours_training * gpu_power_watt
    energy_gpu_inference = inference_gpu_hours * gpu_power_watt

    # Total energy = training + inference + CPU + RAM
    total_energy_wh = energy_cpu + energy_ram + energy_gpu_training + energy_gpu_inference
    total_energy_kwh = total_energy_wh / 1000
    estimated_cost = total_energy_kwh * cost_per_kwh

    return {
        # Provisioning
        "CPU_Cores_Required": total_cpu,
        "RAM_GB_Required": total_ram,
        "Total_CPU_Hours": total_cpu_hours,
        "Total_RAM_Hours": total_ram_hours,
        "Training_GPU_Hours_per_Day": total_gpu_hours_training,
        "Inference_GPU_Hours_per_Day": inference_gpu_hours,

        # Compute estimates
        "Training_Total_FLOPs_per_Day": training_flops,
        "Inference_Total_FLOPs_per_Day": inference_flops_total,

        # Energy usage breakdown (Wh)
        "Energy_CPU_Wh": energy_cpu,
        "Energy_RAM_Wh": energy_ram,
        "Energy_GPU_Training_Wh": energy_gpu_training,
        "Energy_GPU_Inference_Wh": energy_gpu_inference,
        "Total_Energy_Wh": total_energy_wh,
        "Total_Energy_kWh": total_energy_kwh,

        # Cost estimate
        "Estimated_Energy_Cost_USD": estimated_cost
    }
#%%
def estimate_associated_infrastructure_needs(N_servers,
                                             N_bw_per_server=10,
                                             c_rack_per_server=0.5,  # kW/server for cooling
                                             servers_per_rack=20):
    """
    Estimate infrastructure system sizing for power support, cooling, and space (no energy counted).

    Parameters:
    - N_servers: Total number of servers (used to size racks and cooling)
    - N_bw_per_server: Network bandwidth per server in Gbps (default: 10 Gbps)
    - c_rack_per_server: Cooling required per server in kW (default: 0.5 kW/server)
    - servers_per_rack: Number of servers per physical rack (default: 20)

    Returns:
    Dictionary containing:
    - Total cooling requirement (kW)
    - Total network bandwidth (Gbps)
    - Number of racks required
    """
    total_cooling_kw = c_rack_per_server * N_servers
    total_network_bandwidth_gbps = N_bw_per_server * N_servers
    total_racks = (N_servers + servers_per_rack - 1) // servers_per_rack  # Ceiling

    return {
        "Total_Cooling_kW": total_cooling_kw,
        "Total_Network_Bandwidth_Gbps": total_network_bandwidth_gbps,
        "Total_Racks_Required": total_racks
    }
#%%