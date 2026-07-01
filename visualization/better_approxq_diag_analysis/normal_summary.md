# ApproxQ feature diagnostics summary

Performance source: `train_landlord_wp`

## Most impact-aligned features

| rank | position | feature | impact_score | improvement_corr | mean_abs_contribution | mean_abs_td_x_feature |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | landlord | landlord_down_cards_left | 0.712105 | -0.167773 | 0.182067 | 0.102069 |
| 2 | landlord | position_landlord | 0.677237 | -0.038638 | 0.099206 | 0.248912 |
| 3 | landlord | bias | 0.677237 | -0.038638 | 0.099206 | 0.248912 |
| 4 | landlord | hand_singles | 0.671351 | 0.047124 | 0.188658 | 0.042095 |
| 5 | landlord | max_enemy_cards_left | 0.659838 | -0.081240 | 0.163059 | 0.116865 |
| 6 | landlord | unseen_low_cards_ratio | 0.608306 | 0.179784 | 0.154031 | 0.076434 |
| 7 | landlord | unseen_group_low_3_to_7 | 0.608306 | 0.179784 | 0.154031 | 0.076434 |
| 8 | landlord_down | is_farmer | 0.579451 | 0.186999 | 0.078278 | 0.260176 |
| 9 | landlord_down | position_landlord_down | 0.579451 | 0.186999 | 0.078278 | 0.260176 |
| 10 | landlord_down | bias | 0.579451 | 0.186999 | 0.078278 | 0.260176 |

## Most unstable features

| rank | position | feature | instability_score | relative_jitter | direction_changes | sign_flips | total_abs_delta_w |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | landlord_down | action_type_0 | 0.670118 | 0.999943 | 17 | 11267 | 558.679365 |
| 2 | landlord_down | action_is_pass | 0.670118 | 0.999943 | 17 | 11267 | 558.679365 |
| 3 | landlord_up | action_preserves_control | 0.664537 | 0.999963 | 14 | 9121 | 919.529288 |
| 4 | landlord_down | played_group_control_a_2_joker | 0.651168 | 0.999912 | 17 | 7949 | 653.319619 |
| 5 | landlord_down | action_preserves_control | 0.636357 | 0.999989 | 14 | 7050 | 937.306580 |
| 6 | landlord | action_preserves_control | 0.621885 | 0.999922 | 11 | 9461 | 908.170613 |
| 7 | landlord_up | is_farmer | 0.621220 | 0.999979 | 14 | 5523 | 1035.158962 |
| 8 | landlord_up | position_landlord_up | 0.621220 | 0.999979 | 14 | 5523 | 1035.158962 |
| 9 | landlord_up | bias | 0.621220 | 0.999979 | 14 | 5523 | 1035.158962 |
| 10 | landlord_down | played_high_cards_ratio | 0.618689 | 0.999896 | 15 | 7111 | 649.450971 |
