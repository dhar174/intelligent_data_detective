# Visualization Plan for sample_dirty

- Value distribution (histogram): 10 bins
- Score distribution (histogram): 10 bins
- Category distribution (bar chart)
- Scatter: value vs score by category (ASCII table)

ASCII histogram for value (bins 1.0 - 95.0, counts: 6,4,5,3,5,2,5,6,6,5)

Bin 1 (1.0-10.4): ██████
Bin 2 (10.4-19.8): ████
Bin 3 (19.8-29.2): █████
Bin 4 (29.2-38.6): ███
Bin 5 (38.6-48.0): █████
Bin 6 (48.0-57.4): ██
Bin 7 (57.4-66.8): █████
Bin 8 (66.8-76.2): ██████
Bin 9 (76.2-85.6): ██████
Bin 10 (85.6-95.0): █████

ASCII histogram for score (bins 0.1-0.93, counts: 8,7,4,3,1,5,6,7,2,7)

Bin 1 (0.1-0.183): ████████
Bin 2 (0.183-0.266): ███████
Bin 3 (0.266-0.349): ████
Bin 4 (0.349-0.432): ███
Bin 5 (0.432-0.515): █
Bin 6 (0.515-0.598): █████
Bin 7 (0.598-0.681): ███████
Bin 8 (0.681-0.764): ███████
Bin 9 (0.764-0.847): ██
Bin 10 (0.847-0.93): ███████

Category distribution:
- A: 16
- B: 12
- C: 25

Correlation (value vs score): -0.04 (very weak negative)
