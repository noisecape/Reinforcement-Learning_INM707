class Car:

    def __init__(self, start_location):
        self.current_location = start_location

    def drive(self):
        """
        This function handles the movement of the car. At each time step the car moves
        from left to right by 1 location. Whenever it enters within a location with a wall,
        the car will be respawned to the location (row, 1).
        :return:
        """
        self.current_location = self.current_location[0], self.current_location[1]+1
        return self.current_location

    def respawn_car(self):
        """
        This function is used to respawn the car to the location (row, 1)
        :return:
        """
        self.current_location = self.current_location[0], 1
        return self.current_location
