import pygame
from pong import Game
import neat
import os
import pickle


class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai_against_player(self, genome, config):
        """
        Test the AI using against real player.

        """

        # Create a feedforward neural network from the genome and configuration
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Initialize the game loop
        run = True
        clock = pygame.time.Clock()
        while run:
            # Game speed
            clock.tick(125)

            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            # Get the keys pressed by the user
            keys = pygame.key.get_pressed()

            # Move the paddle based on user input
            if keys[pygame.K_z]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            # Activate the neural network with game state inputs
            output = net.activate(
                (self.right_paddle.y, self.ball.y, self.ball.x, abs(self.right_paddle.x - self.ball.x)))
            decision = output.index(max(output))
            # Make decisions based on neural network output
            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            # Update game state and draw the screen
            game_info = self.game.loop()
            self.game.draw(True, False)
            pygame.display.update()

        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        """
        Trains the AI using the provided genomes and configuration.

        Args:
            genome1: Genome of player 1.
            genome2: Genome of player 2.
            config: NEAT configuration.
        """
        # Create neural networks for player 1 and player 2
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            # Player 1 decision making

            output1 = net1.activate(
                (self.left_paddle.y, self.ball.y, self.ball.x, abs(self.left_paddle.x - self.ball.x)))
            decision1 = output1.index(max(output1))

            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)


            # Player 2 decision making
            output2 = net2.activate(
                (self.right_paddle.y, self.ball.y, self.ball.x, abs(self.right_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))

            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()

            # self.game.draw(draw_score=False, draw_hits=True)
            # pygame.display.update()

            # Check game conditions for fitness calculation, end game if any player reached 50 hits
            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break

    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits


def eval_genomes(genomes, config):
    """
    Evaluates the genomes by training AI players against each other using the provided configuration.
    Each genome is trained against all other genomes in the list.
    """
    width, height = 1000, 800
    window = pygame.display.set_mode((width, height))

    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness is None else genome2.fitness
            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)


def run_neat(config):
    """
    Runs the NEAT algorithm with the provided configuration.

    """
    # Initialize the NEAT population

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint99')
    # p = neat.Population(config)

    # Add reporters for NEAT statistics
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    local_dir = os.path.dirname(__file__)
    checkpoint_path = os.path.join(local_dir, r"CheckpointSave\neat-checkpoint")

    p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=checkpoint_path))

    # Run NEAT algorithm to find the winner genome
    winner = p.run(eval_genomes, 200)
    # Save the winning genome to a file
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_ai(config):
    """
    Initializes a game and tests the AI using the provided configuration against real player.

    """
    width, height = 1000, 800
    window = pygame.display.set_mode((width, height))

    # Load the best AI model from the pickle file
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    # Initialize the Pong game and test the AI with the winner model and configuration
    game = PongGame(window, width, height)
    game.test_ai_against_player(winner, config)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # run_neat(config)
    test_ai(config)

