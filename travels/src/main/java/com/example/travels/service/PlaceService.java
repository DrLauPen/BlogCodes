package com.example.travels.service;

import com.example.travels.entity.Place;

import java.util.List;

public interface PlaceService {
    //查找某省份的某一页的景点信息
    List<Place> findByProvinceIdPages(Integer page,
                                      Integer rows,
                                      String provinceId);
    //查找某省份的总景点数目
    Integer findByProvinceIdCounts(String provinceId);

    //保存景点信息
    void save(Place place);

    //删除景点信息
    void deletePlace(String id);

    //找一个景点的信息
    Place findOnePlace(String id);

    //更新景点的信息
    void updatePlace(Place place);
}
