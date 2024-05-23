
def init_washing_plate():
    action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "walk to the dishes",  # 4
            "walk to the dishwasher",  # 5

            "grab the dishes",  # 6

            "put the dishes in the dishwasher",  # 7

            'open the dishwasher',  # 8
            'close the dishwasher',  # 9
        ]
    obs2text = obs2text_washing_plate
    return action_template, obs2text


def obs2text_washing_plate(obs, action_template):
    text = ""

    in_kitchen = obs[0]
    in_bathroom = obs[1]
    in_bedroom = obs[2]
    in_livingroom = obs[3]

    see_dishes = obs[4]
    close_to_dishes = obs[5]
    hold_dishes = obs[6]

    see_dishwasher = obs[7]
    close_to_dishwasher = obs[8]
    is_dishwasher_open = obs[9]

    dishes_in_dishwasher = obs[10]

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

        if not see_dishes:
            object_text += "The dishes is in the dishwasher. "
        else:
            object_text += "You notice dishes and dishwasher. "

        if hold_dishes:
            object_text += "Currently, you have grabbed the dishes in hand. "
            if close_to_dishwasher:
                object_text += "The dishwasher is close to you. "
                action_list = [0, 2, 3, 4, 7, 8, 9]
            else:
                object_text += "The dishwasher is not close to you. "
                action_list = [0, 2, 3, 4, 5]
        else:
            if close_to_dishes and not close_to_dishwasher:
                object_text += "Currently, you are not grabbing anything in hand. The dishes is close to you. "
                action_list = [0, 2, 3, 5, 6]
            elif close_to_dishwasher and not close_to_dishes:
                object_text += "Currently, you are not grabbing anything in hand. The dishwasher is close to you. "
                action_list = [0, 2, 3, 4, 8, 9]
            elif not close_to_dishes and not close_to_dishwasher:
                object_text += "Currently, you are not grabbing anything in hand. The dishes and the dishwasher are not close to you. "
                action_list = [0, 2, 3, 4, 5]
            else:
                if is_dishwasher_open:
                    action_list = [0, 2, 3, 8, 9]
                else:
                    action_list = [0, 2, 3, 9]

        if see_dishes and is_dishwasher_open:
            object_text += "The dishwasher is opened. "
        elif see_dishes and not is_dishwasher_open:
            object_text += "The dishwasher is not opend. "
        else:
            object_text += "The dishwasher is closed. "
            action_list = [0, 2, 3]

    elif in_bathroom:

        if hold_dishes:
            object_text += "and notice nothing useful. Currently, you have grabbed the dishes in hand. "
        else:
            object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

        action_list = [0, 1, 3]
    elif in_bedroom:

        if hold_dishes:
            object_text += "and notice nothing useful. Currently, you have grabbed the dishes in hand. "
        else:
            object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

        action_list = [0, 1, 2]
    elif in_livingroom:

        if hold_dishes:
            object_text += "and notice nothing useful. Currently, you have grabbed the dishes in hand. "
        else:
            object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

        action_list = [1, 2, 3]

    text += object_text

    target_template = "In order to heat up the dishes in the dishwasher, "
    text += target_template

    # template for next step
    next_step_text = "your next step is to"
    text += next_step_text

    actions = [action_template[i] for i in action_list]

    return {"prompt": text, "avaliable_action": actions}