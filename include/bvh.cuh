#pragma once

#include "aabb.cuh"

struct __align__(16) BvhNode {
        Aabb bbox;
        union {
                struct {
                        int idx_l;
                        int idx_r;
                } inner;

                struct {
                        int count;
                        int idx_start;
                } leaf;
        };
        bool is_leaf;
};
