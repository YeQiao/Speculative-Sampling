10/23
implement mamba hindden state cache
optimization
larger nodel
think and quantify overhead of excution
better finetuen, targeting ~70%
benchmark
edge platform: jetson orin nano super


from power_monitor import PowerMonitor
pm = PowerMonitor()
pm.start()
<<<region to moinitor>>>
pm.stop()
print(f"gpt-oss took {pm.get_runtime():.2f} seconds and {pm.get_cpu_energy() + pm.get_gpu_energy()} J energy")
