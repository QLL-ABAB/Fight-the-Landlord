# better_history features (143 dims)

## 中文总览

`better_history` 是裁剪后的 143 维特征版本。它保留位置、剩余牌数、手牌结构、动作上下文、控制牌/炸弹、低中高牌分组摘要，以及地主/农民特化上下文；同时删除了大部分旧的逐牌面历史特征，例如 `my_hand_*`、`played_*`、`total_played_*`、`unseen_*`、`last_move_*`、`last_two_move_*`、`landlord_bottom_*` 和旧 `action_3...action_30`。

这里的“逐牌面”指把 3、4、5、...、A、2、小王、大王分别展开成单独维度。裁剪版的主要目的，是把这些高方差、容易随单张牌偶然出现而抖动的特征，替换成更平滑的分组和角色上下文特征。

## action_context (18)
说明：候选动作本身的描述，包括是否 pass、是否炸弹、是否出完、动作牌型、动作大小，以及动作是否使用/保留控制牌。这类特征直接决定 Q(s,a) 中“这手牌值不值得出”，也是 pass 和压制策略最敏感的部分。

23:action_cards, 24:action_is_pass, 25:action_is_nonpass, 26:action_is_bomb, 27:action_is_king_bomb, 28:action_finish, 31:action_min_rank, 32:action_max_rank, 33:action_avg_rank, 34:action_type_0, 35:action_type_1, 36:action_type_2, 91:action_over_last_rank_gap, 92:action_uses_control_card, 93:action_uses_2, 94:action_uses_joker, 95:action_uses_low_card, 96:action_preserves_control

## cards_left (8)
说明：三家当前剩余牌数、队友剩余牌数、敌人最少/最多剩余牌数，以及出完候选动作后自己还剩多少牌。它刻画局面阶段和危险程度，是判断要不要抢先手、压地主、放队友的重要基础。

5:my_cards_left, 6:landlord_cards_left, 7:landlord_up_cards_left, 8:landlord_down_cards_left, 9:teammate_cards_left, 10:min_enemy_cards_left, 11:max_enemy_cards_left, 29:next_cards_left

## control_bomb (10)
说明：炸弹、王、2、A 等控制资源相关特征，描述当前是否有炸弹、出牌后控制牌变化、是否把炸弹用在临近收尾，以及自己控制牌相对未见牌/已出牌的优势。这类特征重要但也容易抖，需要和局面阶段一起看。

16:bomb_num, 51:bombs_delta, 52:control_delta, 59:bomb_not_near_finish, 60:bomb_near_finish, 75:my_group_control_a_2_joker, 97:control_cards_after_action, 98:control_delta_ratio, 99:my_control_advantage_over_unseen, 100:my_control_advantage_over_played

## farmer_role_specialized (20)
说明：农民协作特征，描述地主是否危险、队友是否快走、pass 是让队友还是放地主、当前动作是否压队友/压地主、是否保留控制牌对抗地主，以及地主上家/下家的座位职责差异。这是裁剪版相对原始 approxq 最关键的角色特化部分。

123:farmer_landlord_left_le_1, 124:farmer_landlord_left_le_2, 125:farmer_landlord_left_le_3, 126:farmer_teammate_left_le_1, 127:farmer_teammate_left_le_2, 128:farmer_teammate_left_le_3, 129:farmer_teammate_about_to_win, 130:farmer_landlord_about_to_win, 131:farmer_pass_to_teammate, 132:farmer_pass_to_landlord, 133:farmer_action_blocks_teammate, 134:farmer_action_blocks_landlord, 135:farmer_save_control_for_landlord, 136:farmer_control_advantage, 137:up_before_landlord_action_is_pass, 138:up_before_landlord_action_is_control, 139:up_should_block_landlord, 140:down_after_landlord_action_is_pass, 141:down_follow_landlord_pressure, 142:down_should_take_lead_from_landlord

## hand_structure (6)
说明：自己当前手牌结构，包括单牌、对子、三张、炸弹、控制牌数量和手牌坏度。它不记录具体牌面，而是概括手牌是否散、是否容易走完、是否有控制资源。

17:hand_singles, 18:hand_pairs, 19:hand_triples, 20:hand_bombs, 21:hand_control, 22:hand_badness

## landlord_specialized (9)
说明：地主视角的进攻和防守特征，描述农民是否只剩 1/2/3 张、两个农民是否都很危险、地主是否应该压制、当前动作是否阻断农民出完，以及炸弹是用于收尾还是浪费。

113:landlord_enemy_min_left_le_1, 114:landlord_enemy_min_left_le_2, 115:landlord_enemy_min_left_le_3, 116:landlord_enemy_both_low, 117:landlord_should_press_enemy, 118:landlord_action_blocks_enemy_finish, 119:landlord_bomb_to_finish, 120:landlord_bomb_waste, 122:landlord_control_advantage

## last_move_context (4)
说明：上一手是谁出的、当前是否处于主动出牌轮。它不关心上一手具体牌面，只判断行动关系：自己、队友、敌人，以及当前是不是可以自由领出。

12:leading_round, 13:last_player_self, 14:last_player_teammate, 15:last_player_enemy

## new_grouped_rank_summary (14)
说明：新增的低牌/中牌/A2/王/控制牌分组摘要，覆盖未见牌、已出牌和当前动作。它用粗粒度牌力分组替代旧逐牌面历史，减少 `total_played_*`、`unseen_*` 等单牌面特征带来的高方差。

61:unseen_group_low_3_to_7, 62:unseen_group_mid_8_to_k, 63:unseen_group_ace_2, 64:unseen_group_joker, 65:unseen_group_control_a_2_joker, 66:played_group_low_3_to_7, 67:played_group_mid_8_to_k, 68:played_group_ace_2, 69:played_group_joker, 70:played_group_control_a_2_joker, 77:action_group_mid_8_to_k, 78:action_group_ace_2, 79:action_group_joker, 80:action_group_control_a_2_joker

## old_rank_action_counts (11)
说明：保留的旧动作牌型编码和低牌动作分组。裁剪版已经删除 `action_3...action_30` 这种当前动作逐牌面计数，只保留较抽象的动作类型和分组信息。

37:action_type_3, 38:action_type_4, 39:action_type_5, 40:action_type_6, 41:action_type_7, 42:action_type_8, 43:action_type_9, 44:action_type_10, 45:action_type_11, 46:action_type_12, 76:action_group_low_3_to_7

## old_rank_landlord_bottom (1)
说明：地主底牌相关的保留摘要，只记录底牌中控制牌数量。裁剪版删除了具体底牌牌面 `landlord_bottom_3...landlord_bottom_30`，避免模型过度依赖单张底牌。

121:landlord_bottom_control_count

## old_rank_last_move (8)
说明：上一手牌的摘要信息，包括张数、是否 pass、是否炸弹、大小和候选动作是否同牌型/是否压过。裁剪版删除了上一手的逐牌面计数，只保留能判断压制关系的摘要。

83:last_move_cards, 84:last_move_is_pass, 85:last_move_is_bomb, 86:last_move_min_rank, 87:last_move_max_rank, 88:last_move_avg_rank, 89:last_move_action_same_type, 90:last_move_action_raises_rank

## other_compact_base (21)
说明：原 compact 中保留的通用策略摘要，包括出牌后是否只剩两张以内、手牌坏度/单牌/对子/三张变化、队友/敌人危险状态、自己低中高牌和王的分组比例，以及是否在队友/敌人后获得领出机会。

30:next_le_two, 47:badness_delta, 48:singles_delta, 49:pairs_delta, 50:triples_delta, 53:teammate_danger, 54:teammate_danger_pass, 55:teammate_danger_block, 56:enemy_danger, 57:enemy_danger_press, 58:enemy_danger_pass, 71:my_group_low_3_to_7, 72:my_group_mid_8_to_k, 73:my_group_ace_2, 74:my_group_joker, 81:lead_after_teammate, 82:lead_after_enemy, 109:my_low_cards_ratio, 110:my_mid_cards_ratio, 111:my_high_cards_ratio, 112:my_jokers_ratio

## position_identity (5)
说明：位置和身份特征，包括地主、地主上家、地主下家、是否农民，以及常数 bias。注意模型本身已经按位置分三套权重，这类特征有一定冗余，但能帮助共享特征表达身份差异。

0:bias, 1:position_landlord, 2:position_landlord_up, 3:position_landlord_down, 4:is_farmer

## rank_summary (8)
说明：未见牌和已出牌的低/中/高/王比例摘要。相比逐牌面计数，它更平滑，主要刻画牌局剩余控制力和低牌压力。

101:unseen_low_cards_ratio, 102:unseen_mid_cards_ratio, 103:unseen_high_cards_ratio, 104:unseen_jokers_ratio, 105:played_low_cards_ratio, 106:played_mid_cards_ratio, 107:played_high_cards_ratio, 108:played_jokers_ratio
