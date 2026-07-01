# ApproxQ feature diagnostics summary

Performance source: `train_landlord_wp`

## Most impact-aligned features

| rank | position | feature | impact_score | improvement_corr | mean_abs_contribution | mean_abs_td_x_feature |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | landlord | hand_badness | 0.714578 | -0.016850 | 0.326753 | 0.042983 |
| 2 | landlord | position_landlord | 0.629992 | 0.342491 | 0.134245 | 0.251582 |
| 3 | landlord | bias | 0.629992 | 0.342491 | 0.134245 | 0.251582 |
| 4 | landlord | hand_singles | 0.600791 | -0.091436 | 0.268908 | 0.040349 |
| 5 | landlord | max_enemy_cards_left | 0.595352 | -0.180846 | 0.199679 | 0.130735 |
| 6 | landlord_down | played_control_cards | 0.509066 | 0.345260 | 0.086474 | 0.244149 |
| 7 | landlord_up | played_control_cards | 0.479545 | 0.565647 | 0.056779 | 0.242106 |
| 8 | landlord | min_enemy_cards_left | 0.445065 | -0.087005 | 0.162305 | 0.077731 |
| 9 | landlord | hand_pairs | 0.429425 | 0.115148 | 0.192600 | 0.023983 |
| 10 | landlord | played_bomb_like_ranks | 0.409434 | -0.017366 | 0.152289 | 0.069371 |

## Most unstable features

| rank | position | feature | instability_score | relative_jitter | direction_changes | sign_flips | total_abs_delta_w |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | landlord | played_control_cards | 0.678136 | 0.999936 | 15 | 9971 | 1625.428007 |
| 2 | landlord_up | is_farmer | 0.674261 | 0.999862 | 11 | 19117 | 1590.417932 |
| 3 | landlord_up | position_landlord_up | 0.674261 | 0.999862 | 11 | 19117 | 1590.417932 |
| 4 | landlord_up | bias | 0.674261 | 0.999862 | 11 | 19117 | 1590.417932 |
| 5 | landlord_down | total_played_7 | 0.646844 | 0.999845 | 16 | 7435 | 987.543114 |
| 6 | landlord_up | played_landlord_down_20 | 0.637229 | 0.997694 | 13 | 868 | 246.535036 |
| 7 | landlord_up | total_played_7 | 0.636578 | 0.999696 | 14 | 7415 | 972.008572 |
| 8 | landlord | total_played_7 | 0.636542 | 0.999876 | 18 | 3844 | 1003.102820 |
| 9 | landlord_up | played_control_cards | 0.632186 | 0.999779 | 13 | 5836 | 1534.508589 |
| 10 | landlord | action_finish | 0.631613 | 0.997060 | 13 | 0 | 197.738599 |
