Total benchmark time: 107.59 minutes

======================================================================
FINAL RESULTS SUMMARY
======================================================================

Model      Accuracy             F1 Score
--------------------------------------------------
LSTM       98.82 ± 0.13%        98.81 ± 0.13%
GRU        98.97 ± 0.06%        98.96 ± 0.07%
A-GRU      98.63 ± 0.06%        98.62 ± 0.06%
======================================================================




------------------------------------------------------------
Test Results for A-GRU:
  Loss: 0.2516
  Accuracy: 98.09%
  Precision: 98.10%
  Recall: 98.08%
  F1 Score: 98.09%
------------------------------------------------------------

Total benchmark time: 9522.68 minutes

======================================================================
FINAL RESULTS SUMMARY
======================================================================

Model      Accuracy        F1 Score
----------------------------------------
LSTM       97.39%          97.37%
GRU        98.09%          98.07%
A-GRU      98.09%          98.09%
======================================================================
Saved training curves to: ./results/pixel_by_pixel_mnist/training_curves.png
Saved accuracy curves to: ./results/pixel_by_pixel_mnist/accuracy_curves.png
Saved LSTM individual curves to: ./results/pixel_by_pixel_mnist/individual_model_lstm.png
Saved GRU individual curves to: ./results/pixel_by_pixel_mnist/individual_model_gru.png
Saved A-GRU individual curves to: ./results/pixel_by_pixel_mnist/individual_model_a_gru.png
Saved test comparison to: ./results/pixel_by_pixel_mnist/test_comparison.png
Saved comparison table to: ./results/pixel_by_pixel_mnist/results_summary.md

Results saved to: ./results/pixel_by_pixel_mnist
Benchmark complete!