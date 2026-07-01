# better_history_full features (312 dims)

## 中文总览

`better_history_full` 是未裁剪的 312 维特征版本。它等于原始 history 的 230 维特征，加上新增 82 维分组摘要和地主/农民角色特化特征。

和 `better_history` 裁剪版相比，full 额外保留了大量旧的逐牌面特征：自己手牌逐牌面、三家已出牌逐牌面、总已出牌逐牌面、未见牌逐牌面、上一手/上两手逐牌面、地主底牌逐牌面、当前动作逐牌面。这些特征信息更细，但诊断显示它们更容易带来权重抖动，尤其在较大学习率下会放大不稳定。

## action_context (18)
说明：候选动作本身的描述，包括是否 pass、是否炸弹、是否出完、动作牌型、动作大小，以及动作是否使用/保留控制牌。这类特征直接决定 Q(s,a) 中“这手牌值不值得出”，也是 pass 和压制策略最敏感的部分。

23:action_cards, 24:action_is_pass, 25:action_is_nonpass, 26:action_is_bomb, 27:action_is_king_bomb, 28:action_finish, 31:action_min_rank, 32:action_max_rank, 33:action_avg_rank, 34:action_type_0, 35:action_type_1, 36:action_type_2, 260:action_over_last_rank_gap, 261:action_uses_control_card, 262:action_uses_2, 263:action_uses_joker, 264:action_uses_low_card, 265:action_preserves_control

## cards_left (8)
说明：三家当前剩余牌数、队友剩余牌数、敌人最少/最多剩余牌数，以及出完候选动作后自己还剩多少牌。它刻画局面阶段和危险程度，是判断要不要抢先手、压地主、放队友的重要基础。

5:my_cards_left, 6:landlord_cards_left, 7:landlord_up_cards_left, 8:landlord_down_cards_left, 9:teammate_cards_left, 10:min_enemy_cards_left, 11:max_enemy_cards_left, 29:next_cards_left

## control_bomb (10)
说明：炸弹、王、2、A 等控制资源相关特征，描述当前是否有炸弹、出牌后控制牌变化、是否把炸弹用在临近收尾，以及自己控制牌相对未见牌/已出牌的优势。这类特征重要但也容易抖，需要和局面阶段一起看。

16:bomb_num, 51:bombs_delta, 52:control_delta, 59:bomb_not_near_finish, 60:bomb_near_finish, 244:my_group_control_a_2_joker, 266:control_cards_after_action, 267:control_delta_ratio, 268:my_control_advantage_over_unseen, 269:my_control_advantage_over_played

## farmer_role_specialized (20)
说明：农民协作特征，描述地主是否危险、队友是否快走、pass 是让队友还是放地主、当前动作是否压队友/压地主、是否保留控制牌对抗地主，以及地主上家/下家的座位职责差异。这是 better_approxq 相对原始 approxq 最关键的角色特化部分。

292:farmer_landlord_left_le_1, 293:farmer_landlord_left_le_2, 294:farmer_landlord_left_le_3, 295:farmer_teammate_left_le_1, 296:farmer_teammate_left_le_2, 297:farmer_teammate_left_le_3, 298:farmer_teammate_about_to_win, 299:farmer_landlord_about_to_win, 300:farmer_pass_to_teammate, 301:farmer_pass_to_landlord, 302:farmer_action_blocks_teammate, 303:farmer_action_blocks_landlord, 304:farmer_save_control_for_landlord, 305:farmer_control_advantage, 306:up_before_landlord_action_is_pass, 307:up_before_landlord_action_is_control, 308:up_should_block_landlord, 309:down_after_landlord_action_is_pass, 310:down_follow_landlord_pressure, 311:down_should_take_lead_from_landlord

## hand_structure (6)
说明：自己当前手牌结构，包括单牌、对子、三张、炸弹、控制牌数量和手牌坏度。它不记录具体牌面，而是概括手牌是否散、是否容易走完、是否有控制资源。

17:hand_singles, 18:hand_pairs, 19:hand_triples, 20:hand_bombs, 21:hand_control, 22:hand_badness

## landlord_specialized (9)
说明：地主视角的进攻和防守特征，描述农民是否只剩 1/2/3 张、两个农民是否都很危险、地主是否应该压制、当前动作是否阻断农民出完，以及炸弹是用于收尾还是浪费。

282:landlord_enemy_min_left_le_1, 283:landlord_enemy_min_left_le_2, 284:landlord_enemy_min_left_le_3, 285:landlord_enemy_both_low, 286:landlord_should_press_enemy, 287:landlord_action_blocks_enemy_finish, 288:landlord_bomb_to_finish, 289:landlord_bomb_waste, 291:landlord_control_advantage

## last_move_context (4)
说明：上一手是谁出的、当前是否处于主动出牌轮。它不关心上一手具体牌面，只判断行动关系：自己、队友、敌人，以及当前是不是可以自由领出。

12:leading_round, 13:last_player_self, 14:last_player_teammate, 15:last_player_enemy

## new_grouped_rank_summary (14)
说明：新增的低牌/中牌/A2/王/控制牌分组摘要，覆盖未见牌、已出牌和当前动作。full 版同时保留这些平滑摘要和旧逐牌面历史，因此信息更多，但也更容易出现冗余和相互拉扯。

230:unseen_group_low_3_to_7, 231:unseen_group_mid_8_to_k, 232:unseen_group_ace_2, 233:unseen_group_joker, 234:unseen_group_control_a_2_joker, 235:played_group_low_3_to_7, 236:played_group_mid_8_to_k, 237:played_group_ace_2, 238:played_group_joker, 239:played_group_control_a_2_joker, 246:action_group_mid_8_to_k, 247:action_group_ace_2, 248:action_group_joker, 249:action_group_control_a_2_joker

## old_rank_action_counts (26)
说明：当前候选动作的旧牌型编码和逐牌面计数。`action_3...action_30` 会记录这手牌具体用了哪些牌面，表达更细，但诊断中容易和 `action_min/max/avg_rank`、动作分组特征重复。

37:action_type_3, 38:action_type_4, 39:action_type_5, 40:action_type_6, 41:action_type_7, 42:action_type_8, 43:action_type_9, 44:action_type_10, 45:action_type_11, 46:action_type_12, 211:action_3, 212:action_4, 213:action_5, 214:action_6, 215:action_7, 216:action_8, 217:action_9, 218:action_10, 219:action_11, 220:action_12, 221:action_13, 222:action_14, 223:action_17, 224:action_20, 225:action_30, 245:action_group_low_3_to_7

## old_rank_landlord_bottom (16)
说明：地主底牌的逐牌面信息和底牌控制牌数量。它能给地主视角提供更细的底牌价值信息，但由于底牌只影响地主且样本稀疏，逐牌面维度在训练中容易不稳定。

196:landlord_bottom_3, 197:landlord_bottom_4, 198:landlord_bottom_5, 199:landlord_bottom_6, 200:landlord_bottom_7, 201:landlord_bottom_8, 202:landlord_bottom_9, 203:landlord_bottom_10, 204:landlord_bottom_11, 205:landlord_bottom_12, 206:landlord_bottom_13, 207:landlord_bottom_14, 208:landlord_bottom_17, 209:landlord_bottom_20, 210:landlord_bottom_30, 290:landlord_bottom_control_count

## old_rank_last_move (23)
说明：上一手牌的逐牌面计数加摘要信息。逐牌面部分告诉模型上一手具体是什么牌，摘要部分告诉模型上一手大小、是否炸弹、候选动作是否同牌型/是否压过。full 版信息最完整，但逐牌面上一手在不同牌局中很稀疏。

151:last_move_3, 152:last_move_4, 153:last_move_5, 154:last_move_6, 155:last_move_7, 156:last_move_8, 157:last_move_9, 158:last_move_10, 159:last_move_11, 160:last_move_12, 161:last_move_13, 162:last_move_14, 163:last_move_17, 164:last_move_20, 165:last_move_30, 252:last_move_cards, 253:last_move_is_pass, 254:last_move_is_bomb, 255:last_move_min_rank, 256:last_move_max_rank, 257:last_move_avg_rank, 258:last_move_action_same_type, 259:last_move_action_raises_rank

## old_rank_last_two_moves (30)
说明：上两手牌的逐牌面计数。它保留短期出牌历史，但维度多且稀疏，容易把一次偶然的牌面序列误当成稳定信号；裁剪版完全删除了这一组。

166:last_two_move_0_3, 167:last_two_move_0_4, 168:last_two_move_0_5, 169:last_two_move_0_6, 170:last_two_move_0_7, 171:last_two_move_0_8, 172:last_two_move_0_9, 173:last_two_move_0_10, 174:last_two_move_0_11, 175:last_two_move_0_12, 176:last_two_move_0_13, 177:last_two_move_0_14, 178:last_two_move_0_17, 179:last_two_move_0_20, 180:last_two_move_0_30, 181:last_two_move_1_3, 182:last_two_move_1_4, 183:last_two_move_1_5, 184:last_two_move_1_6, 185:last_two_move_1_7, 186:last_two_move_1_8, 187:last_two_move_1_9, 188:last_two_move_1_10, 189:last_two_move_1_11, 190:last_two_move_1_12, 191:last_two_move_1_13, 192:last_two_move_1_14, 193:last_two_move_1_17, 194:last_two_move_1_20, 195:last_two_move_1_30

## old_rank_my_hand (15)
说明：自己手牌的逐牌面计数。它能表达“具体持有哪些牌”，但和手牌结构、控制牌、低中高分组高度相关。裁剪版删除这组，是为了避免模型过度记忆具体牌面组合。

61:my_hand_3, 62:my_hand_4, 63:my_hand_5, 64:my_hand_6, 65:my_hand_7, 66:my_hand_8, 67:my_hand_9, 68:my_hand_10, 69:my_hand_11, 70:my_hand_12, 71:my_hand_13, 72:my_hand_14, 73:my_hand_17, 74:my_hand_20, 75:my_hand_30

## old_rank_played_landlord (15)
说明：地主已经出过的牌面计数。它描述地主暴露出的具体牌，但逐牌面历史会随牌局偶然顺序剧烈变化，诊断中属于容易放大抖动的旧历史特征。

76:played_landlord_3, 77:played_landlord_4, 78:played_landlord_5, 79:played_landlord_6, 80:played_landlord_7, 81:played_landlord_8, 82:played_landlord_9, 83:played_landlord_10, 84:played_landlord_11, 85:played_landlord_12, 86:played_landlord_13, 87:played_landlord_14, 88:played_landlord_17, 89:played_landlord_20, 90:played_landlord_30

## old_rank_played_landlord_down (15)
说明：地主下家已经出过的牌面计数。它提供一个农民座位的具体出牌历史，但同样稀疏且高方差；裁剪版用已出牌分组摘要替代。

106:played_landlord_down_3, 107:played_landlord_down_4, 108:played_landlord_down_5, 109:played_landlord_down_6, 110:played_landlord_down_7, 111:played_landlord_down_8, 112:played_landlord_down_9, 113:played_landlord_down_10, 114:played_landlord_down_11, 115:played_landlord_down_12, 116:played_landlord_down_13, 117:played_landlord_down_14, 118:played_landlord_down_17, 119:played_landlord_down_20, 120:played_landlord_down_30

## old_rank_played_landlord_up (15)
说明：地主上家已经出过的牌面计数。它提供另一个农民座位的具体出牌历史，信息细但噪声大，和 `played_group_*`、农民角色特化特征有一定重叠。

91:played_landlord_up_3, 92:played_landlord_up_4, 93:played_landlord_up_5, 94:played_landlord_up_6, 95:played_landlord_up_7, 96:played_landlord_up_8, 97:played_landlord_up_9, 98:played_landlord_up_10, 99:played_landlord_up_11, 100:played_landlord_up_12, 101:played_landlord_up_13, 102:played_landlord_up_14, 103:played_landlord_up_17, 104:played_landlord_up_20, 105:played_landlord_up_30

## old_rank_total_played (15)
说明：全局已经出过的牌面计数。它能推断剩余牌力，但诊断里是 full 版最容易抖的类别之一，尤其 `total_played_14/17/20/30` 这类控制牌相关维度会强烈影响 Q 值。

121:total_played_3, 122:total_played_4, 123:total_played_5, 124:total_played_6, 125:total_played_7, 126:total_played_8, 127:total_played_9, 128:total_played_10, 129:total_played_11, 130:total_played_12, 131:total_played_13, 132:total_played_14, 133:total_played_17, 134:total_played_20, 135:total_played_30

## old_rank_unseen (15)
说明：当前视角未见牌的逐牌面计数。它表达隐藏信息估计，但在自博弈中会和出牌历史、控制牌摘要高度耦合；full 版保留它，裁剪版用 `unseen_group_*` 和 `unseen_*_ratio` 替代。

136:unseen_3, 137:unseen_4, 138:unseen_5, 139:unseen_6, 140:unseen_7, 141:unseen_8, 142:unseen_9, 143:unseen_10, 144:unseen_11, 145:unseen_12, 146:unseen_13, 147:unseen_14, 148:unseen_17, 149:unseen_20, 150:unseen_30

## other_compact_base (21)
说明：原 compact 中保留的通用策略摘要，包括出牌后是否只剩两张以内、手牌坏度/单牌/对子/三张变化、队友/敌人危险状态、自己低中高牌和王的分组比例，以及是否在队友/敌人后获得领出机会。

30:next_le_two, 47:badness_delta, 48:singles_delta, 49:pairs_delta, 50:triples_delta, 53:teammate_danger, 54:teammate_danger_pass, 55:teammate_danger_block, 56:enemy_danger, 57:enemy_danger_press, 58:enemy_danger_pass, 240:my_group_low_3_to_7, 241:my_group_mid_8_to_k, 242:my_group_ace_2, 243:my_group_joker, 250:lead_after_teammate, 251:lead_after_enemy, 278:my_low_cards_ratio, 279:my_mid_cards_ratio, 280:my_high_cards_ratio, 281:my_jokers_ratio

## position_identity (5)
说明：位置和身份特征，包括地主、地主上家、地主下家、是否农民，以及常数 bias。注意模型本身已经按位置分三套权重，这类特征有一定冗余，但能帮助共享特征表达身份差异。

0:bias, 1:position_landlord, 2:position_landlord_up, 3:position_landlord_down, 4:is_farmer

## rank_summary (12)
说明：未见牌、已出牌和控制牌/炸弹相关的摘要。full 版比裁剪版多保留 `unseen_possible_bombs`、`unseen_control_cards`、`played_control_cards`、`played_bomb_like_ranks`。其中 `played_control_cards` 在诊断中很重要但也很容易抖。

226:unseen_possible_bombs, 227:unseen_control_cards, 228:played_control_cards, 229:played_bomb_like_ranks, 270:unseen_low_cards_ratio, 271:unseen_mid_cards_ratio, 272:unseen_high_cards_ratio, 273:unseen_jokers_ratio, 274:played_low_cards_ratio, 275:played_mid_cards_ratio, 276:played_high_cards_ratio, 277:played_jokers_ratio
