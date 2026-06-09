#!/usr/bin/env python3
"""
DouZero 终端可视化演示脚本
展示不同智能体之间的斗地主对局
"""

import os
import pickle
import time
from douzero.env.game import GameEnv, EnvCard2RealCard
from douzero.evaluation.simulation import load_card_play_models
import random

# ==================== 配置区域 ====================
AGENT_CONFIG = {
    "landlord": "value",  # 地主智能体
    "landlord_up": "random",  # 地主上家智能体
    "landlord_down": "random",  # 地主下家智能体
}

EVAL_DATA_PATH = "eval_data.pkl"
STEP_DELAY = 1.5
SHOW_INITIAL_HAND = True
# ==================================================


class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def env_cards_to_real(cards):
    """将环境卡牌数字转换为真实卡牌字符串"""
    card_map = {
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "10",
        11: "J",
        12: "Q",
        13: "K",
        14: "A",
        17: "2",
        20: "X",
        30: "D",
    }
    return sorted(
        [card_map.get(c, str(c)) for c in cards],
        key=lambda x: "2345678910JQKAXD".index(x[0]) if x else "",
    )


def format_cards(cards, max_len=50):
    """格式化卡牌显示，限制每行长度"""
    if not cards:
        return "  "

    real_cards = env_cards_to_real(cards)
    # 每张牌占3个字符宽度（如 " 3", "10", " X"）
    formatted = []
    line = ""
    for card in real_cards:
        # 确保每张牌占3个字符
        if len(card) == 1:
            card_str = f" {card} "
        else:
            card_str = f"{card} "

        if len(line) + len(card_str) > max_len:
            formatted.append(line)
            line = card_str
        else:
            line += card_str

    if line:
        formatted.append(line)

    return "\n         ".join(formatted)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_game_state(
    env, step_num, last_action, last_player, initial_hands, three_landlord_cards
):
    """打印游戏状态"""
    clear_screen()

    print(
        f"{Color.CYAN}╔══════════════════════════════════════════════════════════════╗{Color.RESET}"
    )
    print(
        f"{Color.CYAN}║                    DouZero 斗地主可视化演示                    ║{Color.RESET}"
    )
    print(
        f"{Color.CYAN}╚══════════════════════════════════════════════════════════════╝{Color.RESET}"
    )
    print()

    print(
        f"{Color.YELLOW}回合: {step_num:3d} | 炸弹数: {env.bomb_num} | 上一出牌: {last_player}{Color.RESET}"
    )
    print(
        f"{Color.YELLOW}上一手: {format_cards(last_action) if last_action else '  '}{Color.RESET}"
    )
    print()

    # 地主上家
    landlord_up_cards = env.info_sets["landlord_up"].player_hand_cards
    print(
        f"{Color.PURPLE}【地主上家】当前手牌: {len(landlord_up_cards)} 张{Color.RESET}"
    )
    print(f"         {format_cards(landlord_up_cards)}")
    if SHOW_INITIAL_HAND:
        print(
            f"{Color.PURPLE}         初始手牌: {len(initial_hands['landlord_up'])} 张{Color.RESET}"
        )
        print(f"         {format_cards(initial_hands['landlord_up'])}")
    print()

    # 地主
    landlord_cards = env.info_sets["landlord"].player_hand_cards
    print(f"{Color.RED}【  地主  】当前手牌: {len(landlord_cards)} 张{Color.RESET}")
    print(f"         {format_cards(landlord_cards)}")
    if env.three_landlord_cards:
        print(
            f"{Color.RED}         底牌: {format_cards(env.three_landlord_cards)}{Color.RESET}"
        )
    if SHOW_INITIAL_HAND:
        print(
            f"{Color.RED}         初始手牌: {len(initial_hands['landlord'])} 张 (不含底牌){Color.RESET}"
        )
        print(f"         {format_cards(initial_hands['landlord'])}")
        print(
            f"{Color.RED}         底牌: {format_cards(three_landlord_cards)}{Color.RESET}"
        )
    print()

    # 地主下家
    landlord_down_cards = env.info_sets["landlord_down"].player_hand_cards
    print(
        f"{Color.BLUE}【地主下家】当前手牌: {len(landlord_down_cards)} 张{Color.RESET}"
    )
    print(f"         {format_cards(landlord_down_cards)}")
    if SHOW_INITIAL_HAND:
        print(
            f"{Color.BLUE}         初始手牌: {len(initial_hands['landlord_down'])} 张{Color.RESET}"
        )
        print(f"         {format_cards(initial_hands['landlord_down'])}")
    print()

    print(
        f"{Color.CYAN}┌──────────────────────────────────────────────────────────────┐{Color.RESET}"
    )
    print(
        f"{Color.CYAN}│                    出牌历史记录                              │{Color.RESET}"
    )
    print(
        f"{Color.CYAN}└──────────────────────────────────────────────────────────────┘{Color.RESET}"
    )

    positions = ["landlord", "landlord_down", "landlord_up"]
    pos_names = ["地主", "地主下家", "地主上家"]

    for i, action in enumerate(env.card_play_action_seq):
        pos_idx = i % 3
        player_name = pos_names[pos_idx]
        cards = format_cards(action) if action else "  PASS"
        print(f"  [{i+1:2d}] {player_name:8} : {cards}")

    print()

    if not env.game_over:
        current_pos = env.acting_player_position
        current_name = {
            "landlord": "地主",
            "landlord_up": "地主上家",
            "landlord_down": "地主下家",
        }[current_pos]
        print(f"{Color.GREEN}>>> 当前行动: {current_name}{Color.RESET}")


def visualize_game():
    """可视化单局游戏"""
    with open(EVAL_DATA_PATH, "rb") as f:
        card_play_data_list = pickle.load(f)

    index = random.randint(0, len(card_play_data_list) - 1)
    card_play_data = card_play_data_list[index]

    # 保存初始手牌（地主不含底牌，底牌单独保存）
    initial_hands = {
        "landlord": card_play_data["landlord"].copy(),  # 地主初始17张
        "landlord_up": card_play_data["landlord_up"].copy(),  # 农民各17张
        "landlord_down": card_play_data["landlord_down"].copy(),
    }
    three_landlord_cards = card_play_data["three_landlord_cards"].copy()

    # 排序
    for pos in initial_hands:
        initial_hands[pos].sort()
    three_landlord_cards.sort()

    players = load_card_play_models(AGENT_CONFIG)

    env = GameEnv(players)
    env.card_play_init(card_play_data)

    step_num = 0
    last_action = []
    last_player = ""

    print_game_state(
        env, step_num, last_action, last_player, initial_hands, three_landlord_cards
    )
    time.sleep(STEP_DELAY)

    while not env.game_over:
        current_player = env.acting_player_position
        env.step()
        step_num += 1

        if env.card_play_action_seq:
            last_action = env.card_play_action_seq[-1]
            last_player = {
                "landlord": "地主",
                "landlord_up": "地主上家",
                "landlord_down": "地主下家",
            }[current_player]

        print_game_state(
            env, step_num, last_action, last_player, initial_hands, three_landlord_cards
        )
        time.sleep(STEP_DELAY)

    print(
        f"{Color.CYAN}┌──────────────────────────────────────────────────────────────┐{Color.RESET}"
    )
    print(
        f"{Color.CYAN}│                      游戏结束                                │{Color.RESET}"
    )
    print(
        f"{Color.CYAN}└──────────────────────────────────────────────────────────────┘{Color.RESET}"
    )

    winner = env.get_winner()
    bomb_num = env.get_bomb_num()

    if winner == "landlord":
        print(
            f"{Color.RED}🎉 地主获胜！炸弹数: {bomb_num} | 得分: {2 * (2 ** bomb_num)}{Color.RESET}"
        )
    else:
        print(
            f"{Color.GREEN}🎉 农民获胜！炸弹数: {bomb_num} | 得分: {1 * (2 ** bomb_num)}{Color.RESET}"
        )

    print()
    print(f"{Color.YELLOW}对局回放结束，共 {step_num} 回合{Color.RESET}")


def main():
    print(f"{Color.CYAN}=" * 60)
    print(f"  当前玩家配置:")
    print(f"    地主: {AGENT_CONFIG['landlord']}")
    print(f"    地主上家: {AGENT_CONFIG['landlord_up']}")
    print(f"    地主下家: {AGENT_CONFIG['landlord_down']}")
    print(f"  延迟时间: {STEP_DELAY}秒")
    print(f"  显示初始手牌: {'开启' if SHOW_INITIAL_HAND else '关闭'}")
    print(f"=" * 60)
    print(f"{Color.RESET}")

    input("按 Enter 开始演示...")

    visualize_game()


if __name__ == "__main__":
    main()
