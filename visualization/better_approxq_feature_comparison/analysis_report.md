# Better ApproxQ Feature Diagnostics Comparison

## Feature Sets

- better_history: 143 dims.
- better_history_full: 312 dims.
- Trim removes 169 old high-variance history features; no trimmed-only features.

## Removed Feature Categories

- old_rank_action_counts: 15
- old_rank_landlord_bottom: 15
- old_rank_last_move: 15
- old_rank_last_two_moves: 30
- old_rank_my_hand: 15
- old_rank_played_landlord: 15
- old_rank_played_landlord_down: 15
- old_rank_played_landlord_up: 15
- old_rank_total_played: 15
- old_rank_unseen: 15
- rank_summary: 4

## Learning-Rate Summary

|version|lr|final_wp|last5_wp|best_wp|wp_std|mean_td|flips/1k|abs_dw/update|jitter|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|trimmed|10x|0.5320|0.5271|0.5756|0.0186|0.3663|11.998|0.00438607|1.0000|
|trimmed|1x|0.5444|0.5230|0.5444|0.0442|0.2545|1.105|0.000298603|0.9999|
|trimmed|1e-1|0.4056|0.3859|0.4180|0.0278|0.2569|0.139|2.97683e-05|0.9987|
|trimmed|1e-2|0.3084|0.2984|0.4344|0.0499|0.2162|0.026|2.42552e-06|0.9894|
|full|10x|0.2436|0.2746|0.3360|0.0247|0.6498|10.907|0.0066822|1.0000|
|full|1x|0.5304|0.5226|0.5848|0.0339|0.2509|1.697|0.000264264|0.9999|
|full|1e-1|0.5368|0.5129|0.5368|0.0462|0.2681|0.223|2.84036e-05|0.9992|
|full|1e-2|0.4040|0.3800|0.4520|0.0470|0.2177|0.037|2.27178e-06|0.9931|

## Top Features

### trimmed 10x
Instability: bias, action_preserves_control, is_farmer, played_group_control_a_2_joker, played_high_cards_ratio

Importance: bias, action_preserves_control, is_farmer, landlord_down_cards_left, landlord_up_cards_left

### trimmed 1x
Instability: action_preserves_control, played_group_control_a_2_joker, action_type_0, action_is_pass, played_jokers_ratio

Importance: bias, action_preserves_control, played_low_cards_ratio, played_group_low_3_to_7, is_farmer

### trimmed 1e-1
Instability: action_type_0, action_is_pass, played_low_cards_ratio, played_group_low_3_to_7, action_uses_low_card

Importance: bias, action_preserves_control, is_farmer, played_mid_cards_ratio, played_group_mid_8_to_k

### trimmed 1e-2
Instability: action_uses_low_card, action_max_rank, action_type_0, action_is_pass, action_avg_rank

Importance: bias, action_preserves_control, is_farmer, played_jokers_ratio, played_group_joker

### full 10x
Instability: bias, played_control_cards, action_preserves_control, is_farmer, played_group_control_a_2_joker

Importance: action_preserves_control, bias, played_control_cards, is_farmer, total_played_30

### full 1x
Instability: played_control_cards, played_group_control_a_2_joker, played_high_cards_ratio, played_group_ace_2, played_mid_cards_ratio

Importance: bias, played_control_cards, action_preserves_control, is_farmer, total_played_4

### full 1e-1
Instability: played_mid_cards_ratio, played_group_mid_8_to_k, played_control_cards, total_played_13, played_group_control_a_2_joker

Importance: bias, played_control_cards, action_preserves_control, total_played_20, is_farmer

### full 1e-2
Instability: total_played_14, action_avg_rank, played_landlord_down_13, total_played_17, total_played_8

Importance: played_control_cards, bias, action_preserves_control, is_farmer, total_played_30

