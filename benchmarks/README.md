# Benchmarking

This directory contains the primary executable scripts for the project. The `*_benchmark.py` files are used to evaluate model performance, measuring metrics like inference speed and accuracy on the COCO image dataset. The `video_inference_*.py` scripts serve as a practical demonstration, processing an entire video file to produce a new video with pose estimations overlaid. The `_oef` version of the video script additionally implements the One Euro Filter for temporal smoothing to reduce keypoint jitter.

<br>

<br>

# Appendix: The One Euro Filter

## The Core Idea: Separating Signal from Noise 

At its heart, the filter treats your stream of keypoint coordinates as a signal composed of two parts:

* **The True Signal (Low Frequency):** The actual, intended movement of a person's hand or head. This is a relatively slow, smooth motion.
* **The Noise (High Frequency):** The rapid, tiny, back-and-forth oscillations from the model's prediction uncertainty. This is the "jitter."

The classic tool to remove high-frequency noise is a **Low-Pass Filter (LPF)**. It allows low-frequency signals to "pass through" while blocking high-frequency ones. In its simplest form, it's just a weighted average:

`filtered_point = α * new_point + (1 - α) * previous_filtered_point`

Here, `α` (alpha) is the smoothing factor. This leads to the fundamental problem:

* If `α` is **small** (strong smoothing), the filter is great at removing jitter when the point is still, but it will feel slow and **lag** behind when the point moves fast.
* If `α` is **large** (weak smoothing), the filter is very responsive with low lag, but it will fail to remove jitter when the point is still.


<br>

<br>

## The One Euro Filter's Architecture: An Adaptive Solution

The genius of the One Euro Filter is that it makes the smoothing factor `α` **adaptive**. It uses a second filter to measure the signal's speed and adjusts the smoothing in real-time.

Here is the internal architecture:

1.  **The Derivative of the Signal (Velocity):** First, it calculates the rate of change (the derivative) of the incoming signal. This is a noisy estimate of the keypoint's velocity. `velocity = (new_point - previous_point) / time_delta`

2.  **A Filter for Velocity:** The noisy velocity signal is then passed through its own dedicated Low-Pass Filter (the "derivative filter"). This gives a clean, smoothed estimate of the current speed (`edx`). This step is crucial because it prevents a single noisy jump in the signal from making the filter think a fast movement has occurred.

3.  **The Adaptive Loop (The "Magic"):** The smoothed velocity `edx` is now used to control the main filter.
    * It calculates a new, dynamic **cutoff frequency** for the main filter using this formula: `cutoff = min_cutoff + β * abs(edx)`
    * This dynamic cutoff frequency is then used to calculate a new smoothing factor, `α`, for the main signal filter.

4.  **A Filter for the Signal:** Finally, the original, raw signal is passed through its Low-Pass Filter, which now uses the new, dynamically updated `α`.



**In short:** The filter uses a smoothed estimate of the signal's speed to continuously tune itself. When the speed is low, it smooths aggressively. When the speed is high, it automatically becomes more responsive.


<br>

<br>

## The Parameters for an Engineer

* **`min_cutoff` (Hertz):** This is the baseline cutoff frequency for the main signal filter when the velocity is zero. A value of `1.0` means that when a keypoint is perfectly still, the filter will aggressively eliminate any jitter that oscillates faster than 1 time per second (1 Hz). **Lowering this value makes the filter more aggressive on static or slow-moving points.**

* **`beta` (Reactivity/Slope):** This parameter controls *how much* the cutoff frequency increases in response to velocity. It's the "gain" on the velocity feedback. **Lowering this value makes the filter less sensitive to speed changes,** meaning it will continue to apply strong smoothing even as the keypoint starts to move faster.

This two-filter, adaptive architecture is what makes the One Euro Filter so effective at providing clean, jitter-free signals without the disconnected, "laggy" feeling of a simple low-pass filter.
