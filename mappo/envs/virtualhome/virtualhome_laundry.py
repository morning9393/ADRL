
def init_laundry():
    action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "walk to the clothes",  # 4
            "walk to the washing machine",  # 5

            "grab the clothes",  # 6

            "put the clothes in the washing machine",  # 7

            'open the washing machine',  # 8
            'close the washing machine',  # 9
        ]
    obs2text = obs2text_laundry
    return action_template, obs2text


def obs2text_laundry(obs, action_template):
    text = ""

    in_kitchen = obs[0]
    in_bathroom = obs[1]
    in_bedroom = obs[2]
    in_livingroom = obs[3]

    see_clothes = obs[4]
    close_to_clothes = obs[5]
    hold_clothes = obs[6]

    see_washing_machine = obs[7]
    close_to_washing_machine = obs[8]
    is_washing_machine_open = obs[9]

    clothes_in_washing_machine = obs[10]

    assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

    # template for room
    in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {}. "
    if in_kitchen:
        text += in_room_teplate.format("kitchen")
    elif in_bathroom:
        text += in_room_teplate.format("bathroom")
    elif in_bedroom:
        text += in_room_teplate.format("bedroom")
    elif in_livingroom:
        text += in_room_teplate.format("living room")

    object_text = ""
    action_list = []

    if in_kitchen:

        if not see_clothes:
            object_text += "The clothes is in the washing machine. "
        else:
            object_text += "You notice clothes and washing machine. "

        if hold_clothes:
            object_text += "Currently, you have grabbed the clothes in hand. "
            if close_to_washing_machine:
                object_text += "The washing machine is close to you. "
                action_list = [0, 2, 3, 4, 7, 8, 9]
            else:
                object_text += "The washing machine is not close to you. "
                action_list = [0, 2, 3, 4, 5]
        else:
            if close_to_clothes and not close_to_washing_machine:
                object_text += "Currently, you are not grabbing anything in hand. The clothes is close to you. "
                action_list = [0, 2, 3, 5, 6]
            elif close_to_washing_machine and not close_to_clothes:
                object_text += "Currently, you are not grabbing anything in hand. The washing machine is close to you. "
                action_list = [0, 2, 3, 4, 8, 9]
            elif not close_to_clothes and not close_to_washing_machine:
                object_text += "Currently, you are not grabbing anything in hand. The clothes and the washing machine are not close to you. "
                action_list = [0, 2, 3, 4, 5]
            else:
                if is_washing_machine_open:
                    action_list = [0, 2, 3, 8, 9]
                else:
                    action_list = [0, 2, 3, 9]

        if see_clothes and is_washing_machine_open:
            object_text += "The washing machine is opened. "
        elif see_clothes and not is_washing_machine_open:
            object_text += "The washing machine is not opend. "
        else:
            object_text += "The washing machine is closed. "
            action_list = [0, 2, 3]

    elif in_bathroom:

        if hold_clothes:
            object_text += "and notice nothing useful. Currently, you have grabbed the clothes in hand. "
        else:
            object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

        action_list = [0, 1, 3]
    elif in_bedroom:

        if hold_clothes:
            object_text += "and notice nothing useful. Currently, you have grabbed the clothes in hand. "
        else:
            object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

        action_list = [0, 1, 2]
    elif in_livingroom:

        if hold_clothes:
            object_text += "and notice nothing useful. Currently, you have grabbed the clothes in hand. "
        else:
            object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

        action_list = [1, 2, 3]

    text += object_text

    target_template = "In order to heat up the clothes in the washing machine, "
    text += target_template

    # template for next step
    next_step_text = "your next step is to"
    text += next_step_text

    actions = [action_template[i] for i in action_list]

    return {"prompt": text, "avaliable_action": actions}