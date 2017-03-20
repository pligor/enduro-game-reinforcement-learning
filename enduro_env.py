from __future__ import division
import numpy as np


class EnduroEnv(object):
    """note that opponents are ordered from closer to more far away"""

    @staticmethod
    def renderRoad(road):
        for row in road:
            print " - ".join(["{:3d},{:3d}".format(col[0], col[1]) for col in row])

    @staticmethod
    def find_coords(coords, road):
        """the other idea is to split it in four sections and the winner section to even four"""
        prev_row = None
        for rr, row in enumerate(road):
            prev_col = None  # (0,0)
            for cc, col in enumerate(row):
                # (col[1] > coords_opp[1] > prev_col[1]):
                if prev_row is not None and prev_col is not None:

                    # print (prev_row[cc][0], col[0])
                    if (prev_col[0] <= coords[0] <= col[0]) and \
                            (prev_row[cc][1] <= coords[1] <= col[1]):
                        return rr, cc
                #
                # if prev_col is None:
                #
                #     pass
                # else:
                #     if coords_opp[0] > col[0] and prev_col[1] < coords_opp[1] < col[1]:
                #         return rr, cc
                prev_col = col
            prev_row = row

    def detectOpponentsInTheGrid(self, opponents, grid, road):
        #np.savez('../cur_situation.npz', opponents=opponents, grid=grid, road=road)

        for opp in opponents:
            coords_opponent = opp[:2]
            print coords_opponent
            tpl = self.find_coords(coords=coords_opponent, road=road)
            print "loop road {}".format(tpl)

            # xx = 11 - tpl[0]
            # yy = tpl[1]


            # min_xx = max(0, xx - 1)
            # max_xx = min(11, xx + 2)
            # min_yy = max(0, yy - 1)
            # max_yy = min(10, yy + 2)
            # assert np.any(grid[min_xx:max_xx, min_yy:max_yy] == 1)
            # print road[tpl[0]]
            # print road[tpl[0] + 1]
