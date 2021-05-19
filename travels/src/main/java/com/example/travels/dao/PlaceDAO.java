package com.example.travels.dao;

import com.example.travels.entity.Place;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface PlaceDAO extends BaseDAO<Place,String>{
    //查找分页中某一页面的数据
    List<Place> findByProvinceIdPages(@Param("start") Integer start,
                                         @Param("rows") Integer rows,
                                         @Param("provinceId") String provinceId);
    //查找对应的省的所有的数目
    Integer findByProvinceIdCounts(String provinceId);
}
