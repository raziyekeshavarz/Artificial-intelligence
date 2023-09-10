import numpy as np
import random
import copy


class ExperimentGenerator():

    # Provides initial Bord State

    def __init__(self):
        self.initBoardState = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]

    def generateNewProblem(self):
        return self.initBoardState


class Player:

    # Provide Game Status , Legal Moves , Feature Vector , Choose Moves

    def __init__(self, playerSymbol, playerTargetFunctionWeightVector):
        self.playerSymbol = playerSymbol
        self.playerTargetFunctionWeightVector = playerTargetFunctionWeightVector

    def isGameOver(self, board, playerSymbol):

        flag = False
        if board == -1:
            flag = True
        elif ((board[0][0] == board[0][1] == board[0][2] == playerSymbol) or
              (board[1][0] == board[1][1] == board[1][2] == playerSymbol) or
              (board[2][0] == board[2][1] == board[2][2] == playerSymbol) or
              (board[0][0] == board[1][0] == board[2][0] == playerSymbol) or
              (board[0][1] == board[1][1] == board[2][1] == playerSymbol) or
              (board[0][2] == board[1][2] == board[2][2] == playerSymbol) or
              (board[0][0] == board[1][1] == board[2][2] == playerSymbol) or
              (board[0][2] == board[1][1] == board[2][0] == playerSymbol)):
            flag = True
        elif ' ' not in np.array(board).flatten():
            flag = True
        return flag

    def lookForLegalMoves(self, boardState, playerSymbol):

        legalMoves = []
        for i in range(len(boardState[0])):
            for j in range(len(boardState[0])):
                if boardState[i][j] == ' ':
                    tempBoard = copy.deepcopy(boardState)
                    tempBoard[i][j] = playerSymbol
                    legalMoves.append(tempBoard)
        return legalMoves

    def extractFeatures(self, board, playerSymbol1, playerSymbol2):

        w1, w2, w3, w4, w5, w6 = 0, 0, 0, 0, 0, 0
        for i in range(3):

            if (((board[i][0] == playerSymbol1) and (board[i][1] == board[i][2] == ' ')) or
                    ((board[i][2] == playerSymbol1) and (board[i][0] == board[i][1] == ' '))):
                w1 = w1 + 1

            if (((board[0][i] == playerSymbol1) and (board[1][i] == board[2][i] == ' ')) or
                    ((board[2][i] == playerSymbol1) and (board[0][i] == board[1][i] == ' '))):
                w1 = w1 + 1

            if (((board[i][0] == playerSymbol2) and (board[i][1] == board[i][2] == ' ')) or
                    ((board[i][2] == playerSymbol2) and (board[i][0] == board[i][1] == ' '))):
                w2 = w2 + 1

            if (((board[0][i] == playerSymbol2) and (board[1][i] == board[2][i] == ' ')) or
                    ((board[2][i] == playerSymbol2) and (board[0][i] == board[1][i] == ' '))):
                w2 = w2 + 1

            if (((board[i][0] == board[i][1] == playerSymbol1) and ((board[i][2]) == ' ')) or
                    ((board[i][1] == board[i][2] == playerSymbol1) and ((board[i][0]) == ' '))):
                w3 = w3 + 1

            if (((board[i][0] == board[i][1] == playerSymbol2) and ((board[i][2]) == ' ')) or
                    ((board[i][1] == board[i][2] == playerSymbol2) and ((board[i][0]) == ' '))):
                w4 = w4 + 1

            if board[i][0] == board[i][1] == board[i][2] == playerSymbol1:
                w5 = w5 + 1

            if board[i][0] == board[i][1] == board[i][2] == playerSymbol2:
                w6 = w6 + 1

        # W0 = Bias = 1
        feature_vector = [1, w1, w2, w3, w4, w5, w6]
        return feature_vector

    def boardPrint(self, board):

        print('\n')
        print(board[0][0] + '|' + board[0][1] + '|' + board[0][2])
        print("-----")
        print(board[1][0] + '|' + board[1][1] + '|' + board[1][2])
        print("-----")
        print(board[2][0] + '|' + board[2][1] + '|' + board[2][2])
        print('\n')

    def calculateNonFinalBoardScore(self, weight_vector, feature_vector):

        weight_vector = np.array(weight_vector).reshape((len(weight_vector), 1))
        feature_vector = np.array(feature_vector).reshape((len(feature_vector), 1))
        boardScore = np.dot(weight_vector.T, feature_vector)
        return boardScore[0][0]

    def chooseMove(self, board, playerSymbol1, playerSymbol2):

        legalMoves = self.lookForLegalMoves(board, playerSymbol1)
        legalMoveScores = [self.calculateNonFinalBoardScore(self.playerTargetFunctionWeightVector,
                                                            self.extractFeatures(i, playerSymbol1, playerSymbol2)) for i
                           in legalMoves]
        newBoard = legalMoves[np.argmax(legalMoveScores)]
        return (newBoard)

    def chooseRandomMove(self, board, playerSymbol):

        legalMoves = self.lookForLegalMoves(board, playerSymbol)
        newBoard = random.choice(legalMoves)
        return newBoard


class PerformanceSystem:

    # Create Game History For Choosing Random Moves For Players

    def __init__(self, initialBoard, playersTargetFunctionWeightVectors, playerSymbols):
        self.board = initialBoard
        self.playersTargetFunctionWeightVectors = playersTargetFunctionWeightVectors
        self.playerSymbols = playerSymbols

    def isGameOver(self, board, playerSymbol):
        flag = False
        if board == -1:
            flag = True
        elif ((board[0][0] == board[0][1] == board[0][2] == playerSymbol) or
              (board[1][0] == board[1][1] == board[1][2] == playerSymbol) or
              (board[2][0] == board[2][1] == board[2][2] == playerSymbol) or
              (board[0][0] == board[1][0] == board[2][0] == playerSymbol) or
              (board[0][1] == board[1][1] == board[2][1] == playerSymbol) or
              (board[0][2] == board[1][2] == board[2][2] == playerSymbol) or
              (board[0][0] == board[1][1] == board[2][2] == playerSymbol) or
              (board[0][2] == board[1][1] == board[2][0] == playerSymbol)):
            flag = True
        elif ' ' not in np.array(board).flatten():
            flag = True
        return flag

    def generateGameHistory(self):
        gameHistory = []
        gameStatusFlag = True
        player1 = Player(self.playerSymbols[0], self.playersTargetFunctionWeightVectors[0])
        player2 = Player(self.playerSymbols[1], self.playersTargetFunctionWeightVectors[1])
        tempBoard = copy.deepcopy(self.board)
        while gameStatusFlag:
            tempBoard = player1.chooseMove(tempBoard, player1.playerSymbol, player2.playerSymbol)
            gameHistory.append(tempBoard)
            gameStatusFlag = not self.isGameOver(tempBoard, player1.playerSymbol)
            if gameStatusFlag == False:
                break
            tempBoard = player2.chooseRandomMove(tempBoard, player2.playerSymbol)
            gameHistory.append(tempBoard)
            gameStatusFlag = not self.isGameOver(tempBoard, player2.playerSymbol)
        return gameHistory


class Critic:

    def __init__(self, gameHistory):
        self.gameHistory = gameHistory

    def extractFeatures(self, board, playerSymbol1, playerSymbol2):

        w1, w2, w3, w4, w5, w6 = 0, 0, 0, 0, 0, 0
        for i in range(3):

            if (((board[i][0] == playerSymbol1) and (board[i][1] == board[i][2] == ' ')) or
                    ((board[i][2] == playerSymbol1) and (board[i][0] == board[i][1] == ' '))):
                w1 = w1 + 1

            if (((board[0][i] == playerSymbol1) and (board[1][i] == board[2][i] == ' ')) or
                    ((board[2][i] == playerSymbol1) and (board[0][i] == board[1][i] == ' '))):
                w1 = w1 + 1

            if (((board[i][0] == playerSymbol2) and (board[i][1] == board[i][2] == ' ')) or
                    ((board[i][2] == playerSymbol2) and (board[i][0] == board[i][1] == ' '))):
                w2 = w2 + 1

            if (((board[0][i] == playerSymbol2) and (board[1][i] == board[2][i] == ' ')) or
                    ((board[2][i] == playerSymbol2) and (board[0][i] == board[1][i] == ' '))):
                w2 = w2 + 1

            if (((board[i][0] == board[i][1] == playerSymbol1) and ((board[i][2]) == ' ')) or
                    ((board[i][1] == board[i][2] == playerSymbol1) and ((board[i][0]) == ' '))):
                w3 = w3 + 1

            if (((board[i][0] == board[i][1] == playerSymbol2) and ((board[i][2]) == ' ')) or
                    ((board[i][1] == board[i][2] == playerSymbol2) and ((board[i][0]) == ' '))):
                w4 = w4 + 1

            if board[i][0] == board[i][1] == board[i][2] == playerSymbol1:
                w5 = w5 + 1

            if board[i][0] == board[i][1] == board[i][2] == playerSymbol2:
                w6 = w6 + 1

        feature_vector = [1, w1, w2, w3, w4, w5, w6]
        return feature_vector

    def calculateNonFinalBoardScore(self, weight_vector, feature_vector):

        weight_vector = np.array(weight_vector).reshape((len(weight_vector), 1))
        feature_vector = np.array(feature_vector).reshape((len(feature_vector), 1))
        boardScore = np.dot(weight_vector.T, feature_vector)
        return boardScore[0][0]

    def calculateFinalBoardScore(self, board, playerSymbol1, playerSymbol2):

        # Game Have No Winners
        score = 0
        # If Player_1 Wins
        if ((board[0][0] == board[0][1] == board[0][2] == playerSymbol1) or
                (board[1][0] == board[1][1] == board[1][2] == playerSymbol1) or
                (board[2][0] == board[2][1] == board[2][2] == playerSymbol1) or
                (board[0][0] == board[1][0] == board[2][0] == playerSymbol1) or
                (board[0][1] == board[1][1] == board[2][1] == playerSymbol1) or
                (board[0][2] == board[1][2] == board[2][2] == playerSymbol1) or
                (board[0][0] == board[1][1] == board[2][2] == playerSymbol1) or
                (board[0][2] == board[1][1] == board[2][0] == playerSymbol1)):
            score = 100
        # If player_1 Lost
        elif ((board[0][0] == board[0][1] == board[0][2] == playerSymbol2) or
              (board[1][0] == board[1][1] == board[1][2] == playerSymbol2) or
              (board[2][0] == board[2][1] == board[2][2] == playerSymbol2) or
              (board[0][0] == board[1][0] == board[2][0] == playerSymbol2) or
              (board[0][1] == board[1][1] == board[2][1] == playerSymbol2) or
              (board[0][2] == board[1][2] == board[2][2] == playerSymbol2) or
              (board[0][0] == board[1][1] == board[2][2] == playerSymbol2) or
              (board[0][2] == board[1][1] == board[2][0] == playerSymbol2)):
            score = -100
        return score

    def generateTrainingSamples(self, weight_vector, playerSymbol1, playerSymbol2):

        trainingExamples = []
        for i in range(len(self.gameHistory) - 1):
            feature_vector = self.extractFeatures(self.gameHistory[i + 1], playerSymbol1, playerSymbol2)
            trainingExamples.append([feature_vector, self.calculateNonFinalBoardScore(weight_vector, feature_vector)])
        trainingExamples.append([self.extractFeatures(self.gameHistory[-1], playerSymbol1, playerSymbol2),
                                 self.calculateFinalBoardScore(self.gameHistory[-1], playerSymbol1, playerSymbol2)])
        return trainingExamples

    def arrayPrint(self, board):

        print('\n')
        print(board[0][0] + '|' + board[0][1] + '|' + board[0][2])
        print("-----")
        print(board[1][0] + '|' + board[1][1] + '|' + board[1][2])
        print("-----")
        print(board[2][0] + '|' + board[2][1] + '|' + board[2][2])
        print('\n')

    def boardDisplay(self, playerSymbol1, playerSymbol2, gameStatusCount):

        for board in self.gameHistory:
            self.arrayPrint(board)
        finalScore = self.calculateFinalBoardScore(self.gameHistory[-1], playerSymbol1, playerSymbol2)
        if finalScore == 100:
            print(playerSymbol1 + " wins")
            gameStatusCount[0] = gameStatusCount[0] + 1
        elif finalScore == -100:
            print(playerSymbol2 + " wins")
            gameStatusCount[1] = gameStatusCount[1] + 1
        else:
            print("Draw")
            gameStatusCount[2] = gameStatusCount[2] + 1
        return gameStatusCount


class Generalizer:

    def __init__(self, trainingExamples):
        self.trainingExamples = trainingExamples

    def calculateNonFinalBoardScore(self, weight_vector, feature_vector):
        weight_vector = np.array(weight_vector).reshape((len(weight_vector), 1))
        feature_vector = np.array(feature_vector).reshape((len(feature_vector), 1))
        boardScore = np.dot(weight_vector.T, feature_vector)
        return boardScore[0][0]

    def lmsWeightUpdate(self, weight_vector, alpha=0.4):
        for trainingExample in self.trainingExamples:
            vTrainBoardState = trainingExample[1]
            vHatBoardState = self.calculateNonFinalBoardScore(weight_vector, trainingExample[0])
            weight_vector = weight_vector + (alpha * (vTrainBoardState - vHatBoardState) * np.array(trainingExample[0]))
        return weight_vector


def train(numTrainingSamples=3000):
    # Training Sample
    trainingGameCount = 0
    playerSymbols = ('X', 'O')
    playersTargetFunctionWeightVectors = [np.array([.5, .5, .5, .5, .5, .5, .5]),
                                          np.array([.5, .5, .5, .5, .5, .5, .5])]
    gameStatusCount = [0, 0, 0]

    while trainingGameCount < numTrainingSamples:
        # Experiment Generator
        experimentGenerator = ExperimentGenerator()
        initialBoardState = experimentGenerator.generateNewProblem()

        # Performance System
        performanceSystem = PerformanceSystem(initialBoardState, playersTargetFunctionWeightVectors, playerSymbols)
        gameHistory = performanceSystem.generateGameHistory()

        # Critic
        critic = Critic(gameHistory)
        trainingExamplesPlayer1 = critic.generateTrainingSamples(playersTargetFunctionWeightVectors[0],
                                                                 playerSymbols[0], playerSymbols[1])
        trainingExamplesPlayer2 = critic.generateTrainingSamples(playersTargetFunctionWeightVectors[1],
                                                                 playerSymbols[1], playerSymbols[0])

        # Display board states
        gameStatusCount = critic.boardDisplay(playerSymbols[0], playerSymbols[1], gameStatusCount)

        # Generalizer
        generalizer = Generalizer(trainingExamplesPlayer1)
        playersTargetFunctionWeightVectors = [generalizer.lmsWeightUpdate(playersTargetFunctionWeightVectors[0]),
                                              generalizer.lmsWeightUpdate(playersTargetFunctionWeightVectors[1])]

        trainingGameCount = trainingGameCount + 1

    print("\nTraining Results: (" + "Player-1 Wins = " + str(gameStatusCount[0]) +
          ", Player-2 Wins = " + str(gameStatusCount[1]) + ", Game Draws = " + str(gameStatusCount[2]) +
          ")\n")

    # Weight Learn from previous games
    learntWeight = list(np.mean(np.array([playersTargetFunctionWeightVectors[0],
                                          playersTargetFunctionWeightVectors[1]]), axis=0))
    print("Final Learn Weight Vector: \n" + str(learntWeight))

    # AI vs User
    print("\nDo you want to play(y/n) v/s  AI")
    ans = input()
    while ans == "y":

        experimentGenerator = ExperimentGenerator()
        boardState = experimentGenerator.generateNewProblem()
        gameStatusFlag = True
        computer = Player('X', learntWeight)
        gameHistory = []

        print('\nBegin AI(X) v/s User(O) Tic-Tac-Toe\n')
        while gameStatusFlag:

            boardState = computer.chooseMove(boardState, computer.playerSymbol, 'H')
            print('AI\'s Turn:\n')
            computer.boardPrint(boardState)
            gameHistory.append(boardState)
            gameStatusFlag = not computer.isGameOver(boardState, computer.playerSymbol)
            if not gameStatusFlag:
                break

            print('User\'s Turn:\n')
            print('Choose X-coordinate(0-2):')
            x = int(input())
            print('Choose Y-coordinate(0-2):')
            y = int(input())

            boardState[x][y] = 'O'
            computer.boardPrint(boardState)
            gameHistory.append(boardState)
            gameStatusFlag = not computer.isGameOver(boardState, 'H')

        print("Continue playing...? (y/n).")
        ans = input()
        if ans != 'y':
            break


train(7000)
