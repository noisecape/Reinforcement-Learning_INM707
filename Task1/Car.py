class Car:

    def __init__(self, start_location):
        self.current_location = start_location

    def drive(self):
        """
        This function handles the movement of the car. At each time step the car moves
        from left to right. Whenever the wall on the right handside of the board is entered,
        the car will be respawned to the location (row, 1), or in other words the beginning of the road.
        :param new_location: the new location the move will enter.
        :return:
        """
        pass
        # new_location = (self.current_location[0]+1, self.current_location[1])
        # if new_location == EnvironmentUtils.WALL:
        #     self.current_location = (new_location[0], 1)
        # else:
        #     self.current_location = new_location
        # return self.current_location
