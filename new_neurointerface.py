import time
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import socket

from CapsuleSDK.Capsule import Capsule
from CapsuleSDK.DeviceLocator import DeviceLocator
from CapsuleSDK.DeviceType import DeviceType
from CapsuleSDK.Device import Device
from CapsuleSDK.EEGTimedData import EEGTimedData

from eeg_utils import *

# Конфиг
PLATFORM = 'win'
EEG_WINDOW_SECONDS = 4.0
CHANNELS = 2
BUFFER_LEN = int(SAMPLE_RATE * EEG_WINDOW_SECONDS)
TARGET_SERIAL = "821733"

#  Настройки машинки и порога 
ESP32_IP = "10.184.31.76"  #  замените на IP вашей ESP32
UDP_PORT = 9999            #  должен совпадать с main.py на ESP32
THRESHOLD = 2e-11     # порог мощности (подстройте под данные)
CALIBRATION_DURATION = 10.0  # секунд "тишины" при старте

# UDP-сокет для управления
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_to_esp32(cmd):
    """Отправляет команду на ESP32."""
    try:
        udp_sock.sendto((cmd + "\n").encode(), (ESP32_IP, UDP_PORT))
    except OSError as e:
        print(f"[UDP] {e}")

# Инициализация
device = None
device_locator = None
calibration_start_time = None
is_calibrated = False
current_direction = "S"  # "F", "B", "S"


class EventFiredState:
    def __init__(self): self._awake = False
    def is_awake(self): return self._awake
    def set_awake(self): self._awake = True
    def sleep(self): self._awake = False

device_list_event = EventFiredState()
device_conn_event = EventFiredState()
device_eeg_event = EventFiredState()

ring = RingBuffer(n_channels=CHANNELS, maxlen=BUFFER_LEN)
channel_names = []

def send_car_command(cmd):
    """Отправляет команду на ESP32."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        full_cmd = cmd + "\n"
        sock.sendto(full_cmd.encode('utf-8'), (ESP32_IP, UDP_PORT))
    except OSError as e:
        print(f"[CAR] Ошибка: {e}")
    finally:
        sock.close()

def non_blocking_cond_wait(wake_event, name, total_sleep_time):
    print(f"Waiting {name} up to {total_sleep_time}s...")
    steps = int(total_sleep_time * 50)
    for _ in range(steps):
        if device_locator is not None:
            try:
                device_locator.update()
            except:
                pass
        if wake_event.is_awake():
            return True
        time.sleep(0.02)
    return False

def on_device_list(locator, info, fail_reason):
    global device
    chosen = None
    if not info:
        print("No devices found.")
        return
    print(f"Found {len(info)} devices.")
    if TARGET_SERIAL is None:
        chosen = info[0]
    else:
        for dev in info:
            if dev.get_serial() == TARGET_SERIAL:
                chosen = dev
                break
    if not chosen:
        print(f"Target device {TARGET_SERIAL} not found!")
        return
    print(f"Connecting to: {chosen.get_name()} ({chosen.get_serial()})")
    device = Device(locator, chosen.get_serial(), locator.get_lib())
    device_list_event.set_awake()

def on_connection_status_changed(d, status):
    global channel_names
    ch_obj = device.get_channel_names()
    channel_names = [ch_obj.get_name_by_index(i) for i in range(len(ch_obj))]
    print(f"Channel names: {channel_names}")
    device_conn_event.set_awake()

rt_filter = RealTimeFilter(sfreq=250, l_freq=8, h_freq=10, n_channels=CHANNELS)

def on_eeg(d, eeg: EEGTimedData):
    global ring
    samples = eeg.get_samples_count()
    ch = eeg.get_channels_count()
    if samples <= 0: return

    block = np.zeros((ch, samples), dtype=float)
    for i in range(samples):
        for c in range(ch):
            block[c, i] = eeg.get_processed_value(c, i)
    # Фильтрация — один раз, после сбора блока
    filtered_block = rt_filter.filter_block(block)
    
    if filtered_block.shape[0] >= CHANNELS:
        ring.append_block(filtered_block[:CHANNELS, :])
    else:
        padded = np.zeros((CHANNELS, filtered_block.shape[1]), dtype=float)
        padded[:filtered_block.shape[0], :] = filtered_block
        ring.append_block(padded)
    if not device_eeg_event.is_awake():
        device_eeg_event.set_awake()

# Порог и настройки прогресса
ALPHA_LOW = 8.0
ALPHA_HIGH = 12.0

ALPHA_THRESHOLD = 8e-12  # ← настройте под ваши данные
CALIBRATION_DURATION = 10.0  # секунд до старта прогресса
progress_value = 0  # 0..100
last_accum_time = None
progress_step_sec = 0.1  # шаг обновления прогресса

# Замените историю: теперь одна линия — среднее
alpha_avg_history = []  # вместо alpha_history = [[] for _ in range(CHANNELS)]
time_history = []
first_time = None
move = 'STOP'

# Отрисовка на одном графике
# Отрисовка ЭЭГ,PSD, alpha power на трех графиках
fig, (ax_eeg, ax_psd, ax_alpha, ax_progress) = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

# Alpha: ОДНА линия — среднее по каналам
line_alpha_avg, = ax_alpha.plot([], [], 'b-', lw=2, label='Avg Alpha')
# Серая пунктирная линия порога
thr_line = ax_alpha.axhline(ALPHA_THRESHOLD, color='gray', linestyle='--', linewidth=1, label='Threshold')
ax_alpha.set_xlabel("Time (s)") 
ax_alpha.set_ylabel("Alpha Power (µV²/Hz)")
ax_alpha.set_title("Average Alpha Power (8–12 Hz)")
ax_alpha.set_ylim(0, 1e-11)
ax_alpha.legend(loc='upper right')
ax_alpha.grid(True)

lines_eeg = []
for i in range(CHANNELS):
    ln, = ax_eeg.plot([], [], label=f'Ch{i}', lw=1)
    lines_eeg.append(ln)
ax_eeg.set_ylabel("Amplitude (µV)")
ax_eeg.set_title("EEG Channels")
ax_eeg.legend(loc='upper right')
ax_eeg.grid(True)

lines_psd = []
for i in range(CHANNELS):
    ln, = ax_psd.plot([], [], label=f'PSD Ch{i}', lw=1)
    lines_psd.append(ln)
ax_psd.set_xlabel("Frequency (Hz)") # Ось X для PSD
ax_psd.set_ylabel("PSD (µV²/Hz)")
ax_psd.set_title("PSD Channels")
ax_psd.legend(loc='upper right')
ax_psd.grid(True)

# рогресс-бар (4-й график)
ax_progress.set_xlim(0, 100)
ax_progress.set_ylim(-0.5, 0.5)
ax_progress.set_xlabel("Progress (%)")
ax_progress.set_ylabel("")
ax_progress.set_title("Alpha > Threshold → +1 every 0.1s (after 10s)")
bar_container = ax_progress.barh([0], [0], height=0.8, color='tab:green', alpha=0.8)
progress_bar = bar_container[0]
ax_progress.set_yticks([])
ax_progress.grid(True, axis='x')

def old_update_plot(_):
    global channel_names, calibration_start_time, is_calibrated, current_direction

    try:
        device_locator.update()
    except Exception as e:
        print(f"[SDK] Update error: {e}")

    buf = ring.get()
    current_time = time.time()

    if buf.shape[1] == 0:
        return lines_eeg + lines_psd

    # Инициализация времени калибровки
    if calibration_start_time is None:
        calibration_start_time = current_time

    # Обновление EEG
    t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
    for i in range(CHANNELS):
        lines_eeg[i].set_data(t, buf[i, :])
        label = channel_names[i] if i < len(channel_names) else f'Ch{i}'
        lines_eeg[i].set_label(label)
    all_eeg = buf.flatten()
    ymin, ymax = all_eeg.min(), all_eeg.max()
    if ymin == ymax:
        ymin -= 1e-6; ymax += 1e-6
    pad = 0.1 * (ymax - ymin)
    ax_eeg.set_ylim(ymin - pad, ymax + pad)
    ax_eeg.set_xlim(-EEG_WINDOW_SECONDS, 0)
    ax_eeg.legend(loc='upper right')

    # Обновление PSD
    try:
        freqs, psd = compute_psd_mne(buf, sfreq=SAMPLE_RATE, fmin=1.0, fmax=50.0)
        num_ch = min(psd.shape[0], CHANNELS)
        for i in range(num_ch):
            lines_psd[i].set_data(freqs, psd[i, :])
            label = channel_names[i] if i < len(channel_names) else f'PSD Ch{i}'
            lines_psd[i].set_label(label)
        ax_psd.legend(loc='upper right')
    except Exception as e:
        print(f"[PSD] Error: {e}")
        return lines_eeg + lines_psd

    # Управление машинкой
    elapsed = current_time - calibration_start_time
    if not is_calibrated and elapsed >= CALIBRATION_DURATION:
        is_calibrated = True
        print("Калибровка завершена. Машинка готова к управлению.")

    if is_calibrated:
        # Решение: вперёд или назад
        if not (psd > THRESHOLD).any():
            if current_direction != "F":
                send_to_esp32("F,50")
                current_direction = "F"
        else:
            if current_direction != "B":
                send_to_esp32("B,50")
                current_direction = "B"

    return lines_eeg + lines_psd

def update_plot(_):
    global alpha_avg_history, time_history, progress_value, last_accum_time
    global channel_names, calibration_start_time, is_calibrated, current_direction, first_time, move

    try:
        device_locator.update()
    except Exception as e:
        print(f"[SDK] Update error: {e}")

    current_time = time.time()
    buf = ring.get()

    if buf.shape[1] != 0:

        if calibration_start_time is None:
            calibration_start_time = current_time

        t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
        for i in range(CHANNELS):
            lines_eeg[i].set_data(t, buf[i, :])
            if len(channel_names) > i:
                lines_eeg[i].set_label(channel_names[i])
            else:
                lines_eeg[i].set_label(f'Ch{i}')

        all_data_eeg = buf.flatten()
        ymin, ymax = all_data_eeg.min(), all_data_eeg.max()
        if ymin == ymax:
            ymin -= 1e-6; ymax += 1e-6
        pad = 0.1*(ymax - ymin)
        ax_eeg.set_ylim(ymin-pad, ymax+pad)
        ax_eeg.set_xlim(-EEG_WINDOW_SECONDS, 0)
        ax_eeg.legend(loc='upper right')

        # Вычисление PSD из буфера
        try:
            freqs, psd = compute_psd_mne(buf, sfreq=SAMPLE_RATE, fmin=1.0, fmax=50.0)
            num_ch = min(psd.shape[0], CHANNELS)
            for i in range(num_ch):
                lines_psd[i].set_data(freqs, psd[i, :])
                label = channel_names[i] if i < len(channel_names) else f'PSD Ch{i}'
                lines_psd[i].set_label(label)
            ax_psd.legend(loc='upper right')
        except Exception as e:
            print(f"[PSD] Error: {e}")
            return lines_eeg + lines_psd

        current_alpha = []
        for i in range(num_ch):
            alpha_pow = integrate_band(freqs, psd[i, :], ALPHA_LOW, ALPHA_HIGH)
            current_alpha.append(alpha_pow)
        avg_alpha = np.mean(current_alpha) if current_alpha else 0.0

        time_history.append(current_time)
        alpha_avg_history.append(avg_alpha)

        MAX_HISTORY = 100
        if len(time_history) > MAX_HISTORY:
            time_history[:] = time_history[-MAX_HISTORY:]
            alpha_avg_history[:] = alpha_avg_history[:][-MAX_HISTORY:]

        # Обновляем график средней альфа
        if time_history:
            t0 = time_history[0]
            t_rel = [t - t0 for t in time_history]
            line_alpha_avg.set_data(t_rel, alpha_avg_history)
            ax_alpha.set_xlim(t_rel[0], t_rel[-1])

        # Прогресс-бар: +1/-1 каждые 0.1 сек, но только после 10 сек
        if len(time_history) > 0 and (current_time - time_history[0]) >= CALIBRATION_DURATION:
            now = time.time()
            if last_accum_time is None:
                last_accum_time = now
            if now - last_accum_time >= progress_step_sec:
                if avg_alpha > ALPHA_THRESHOLD:
                    progress_value = min(100, progress_value + 50)
                    if not first_time:
                        first_time = now

                    if now - first_time >= 3:
                        move = "F" if move == "B" else "B"

                    if now - first_time >= 5:
                        move = "STOP"
                    
                else:
                    progress_value = max(0, progress_value - 50)
                    if first_time and now - first_time > 0.5:
                        first_time = None
                last_accum_time = now
            progress_bar.set_width(progress_value)
        else:
            progress_bar.set_width(0)  # до 10 сек — ноль

        
           # Управление машинкой
        elapsed = current_time - calibration_start_time
        if not is_calibrated and elapsed >= CALIBRATION_DURATION:
            is_calibrated = True
            print("Калибровка завершена. Машинка готова к управлению.")

        if is_calibrated:
            # Решение: вперёд или назад
            if move == "F":
                send_to_esp32("F,50")
                current_direction = "F"
            if move == "B":
                send_to_esp32("B,50")
                current_direction = "B"
            else:
                send_to_esp32("S")


        return lines_eeg + lines_psd + [line_alpha_avg] + [progress_bar]


def main():
    global device_locator, device
    if PLATFORM == 'win':
        capsuleLib = Capsule('./CapsuleClient.dll')
    else:
        capsuleLib = Capsule('./libCapsuleClient.dylib')

    device_locator = DeviceLocator(capsuleLib.get_lib())
    device_locator.set_on_devices_list(on_device_list)
    device_locator.request_devices(device_type=DeviceType.Band, seconds_to_search=10)

    if not non_blocking_cond_wait(device_list_event, 'device list', 12):
        print("No device found. Exiting.")
        return

    device.set_on_connection_status_changed(on_connection_status_changed)
    device.set_on_eeg(on_eeg)
    device.connect(bipolarChannels=True)
    if not non_blocking_cond_wait(device_conn_event, 'device connection', 40):
        print("Failed to connect.")
        return

    device.start()
    print("Device started. Opening plot...")

    # Начальная остановка машинки
    send_to_esp32("S")

    # интерактивный режим
    plt.ion() 

    # Создаём анимацию без фонового потока
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

    # Отображаем окно и не блокируем основной поток
    plt.show()

    # Основной цикл: ждём, пока окно не закроется
    try:
        while plt.fignum_exists(fig.number):  # пока окно открыто
            plt.pause(0.1)  # короткая пауза, чтобы не грузить CPU
    except KeyboardInterrupt:
        pass
    finally:
        send_to_esp32("S")
        udp_sock.close()
        device.stop()
        device.disconnect()
        print("Stopped.")


if __name__ == '__main__':
    main()
