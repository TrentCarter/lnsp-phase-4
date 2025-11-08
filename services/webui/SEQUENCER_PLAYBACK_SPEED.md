# Sequencer Playback Speed - Logarithmic Scaling

## Overview

The sequencer playback speed now supports **0.1x to 1000x** with **logarithmic scaling** for smooth acceleration across the entire range.

## Speed Scale (Logarithmic)

| Slider Position | Playback Speed | Description |
|----------------|----------------|-------------|
| 0% | 0.1x | Slowest (10% speed) |
| 25% | 1.0x | **Normal speed** (real-time) |
| 50% | 10x | 10x faster |
| 75% | 100x | 100x faster |
| 100% | 1000x | **Maximum speed** (1000x faster) |

## Why Logarithmic?

Logarithmic scaling provides **smooth, intuitive control** across a wide speed range:

- **Linear scaling problem**: At 100x max, going from 1x→10x takes the same slider movement as 90x→100x (not intuitive)
- **Logarithmic solution**: Each 25% of slider movement multiplies speed by 10x (exponential growth feels linear)

## Formula

```javascript
// Slider (0-100) → Speed (0.1x-1000x)
speed = 0.1 * 10^(sliderValue / 25)

// Speed (0.1x-1000x) → Slider (0-100)
sliderPosition = 25 * log10(speed / 0.1)
```

## Examples

| Slider | Calculation | Speed |
|--------|-------------|-------|
| 0 | 0.1 × 10^(0/25) = 0.1 × 10^0 | 0.1x |
| 12.5 | 0.1 × 10^(12.5/25) = 0.1 × 10^0.5 | 0.316x |
| 25 | 0.1 × 10^(25/25) = 0.1 × 10^1 | 1.0x |
| 37.5 | 0.1 × 10^(37.5/25) = 0.1 × 10^1.5 | 3.16x |
| 50 | 0.1 × 10^(50/25) = 0.1 × 10^2 | 10x |
| 62.5 | 0.1 × 10^(62.5/25) = 0.1 × 10^2.5 | 31.6x |
| 75 | 0.1 × 10^(75/25) = 0.1 × 10^3 | 100x |
| 87.5 | 0.1 × 10^(87.5/25) = 0.1 × 10^3.5 | 316x |
| 100 | 0.1 × 10^(100/25) = 0.1 × 10^4 | 1000x |

## Usage in HMI

### Sequencer View (http://localhost:6101/sequencer)

**Two Speed Controls**:
1. **Top toolbar slider** - Quick access during playback
2. **Bottom slider** - Large, prominent control

**Display Format**:
- `< 1.0x`: 2 decimal places (e.g., `0.32x`)
- `1.0x - 9.9x`: 1 decimal place (e.g., `5.4x`)
- `≥ 10x`: Whole numbers (e.g., `350x`)

### Settings (⚙️ → Sequencer)

**Default Playback Speed**:
- Range: 0.1x to 1000x
- Default: 1.0x (normal speed)
- Set your preferred starting speed

## Sound Effects at High Speed

At extreme speeds (100x+), sound events will fire rapidly:

- **Music Mode**: Rapid notes create melodic patterns
- **Voice Mode**: Overlapping speech (may be garbled at 1000x)
- **Geiger Counter**: Rapid clicks simulate high radiation
- **Random Mode**: Burst of random tones

**Tip**: For high-speed playback, use **Music** or **Geiger** modes for best audio experience.

## Performance Notes

- **No performance impact**: Speed is simulation-only (no actual compute load)
- **Smooth animation**: 100ms update interval regardless of speed
- **Sound throttling**: Browser may limit audio playback at extreme speeds
- **Memory efficient**: No buffering or pre-calculation required

## Testing

```javascript
// Open browser console on sequencer page

// Test logarithmic scaling
console.log('0% slider:', sliderToSpeed(0));    // 0.1x
console.log('25% slider:', sliderToSpeed(25));  // 1.0x
console.log('50% slider:', sliderToSpeed(50));  // 10x
console.log('75% slider:', sliderToSpeed(75));  // 100x
console.log('100% slider:', sliderToSpeed(100)); // 1000x

// Test inverse function
console.log('0.1x speed:', speedToSlider(0.1));   // 0
console.log('1.0x speed:', speedToSlider(1.0));   // 25
console.log('10x speed:', speedToSlider(10));     // 50
console.log('100x speed:', speedToSlider(100));   // 75
console.log('1000x speed:', speedToSlider(1000)); // 100
```

## Comparison with Previous Linear Scaling

| Feature | Old (Linear to 100x) | New (Logarithmic to 1000x) |
|---------|---------------------|---------------------------|
| Max Speed | 100x | **1000x** (10x higher) |
| Scaling | Piecewise linear/exponential | Pure logarithmic |
| Slider Resolution | Poor at high speeds | **Excellent across all ranges** |
| 1x Position | 50% | **25%** (more room for slow speeds) |
| Intuitiveness | Moderate | **High** (feels linear) |

## Future Enhancements

- [ ] Speed presets (0.5x, 1x, 10x, 100x, 1000x buttons)
- [ ] Keyboard shortcuts (↑/↓ arrows for speed adjust)
- [ ] Speed indicator with color coding (slow=blue, normal=green, fast=yellow, extreme=red)
- [ ] Audio pitch adjustment at high speeds (chipmunk effect)
- [ ] Frame skipping visualization at extreme speeds

## Credits

Logarithmic speed scaling inspired by professional video editing software (Adobe Premiere, DaVinci Resolve) which use similar exponential controls for timeline scrubbing.
