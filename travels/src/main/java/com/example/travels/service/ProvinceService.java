package com.example.travels.service;

import com.example.travels.entity.Province;

import java.util.List;

public interface ProvinceService {
    List<Province> findByPages(Integer start,Integer rows);//参数1当前页 ；参数2，显示条数

    //查询总条数
    Integer findTotals();

    //保存省份
    void saveProvince(Province province);

    //删除省份
    void deleteProvince(String id);

    //根据id查询省份信息
    Province findOne(String id);

    //修改省份信息
    void updateProvince(Province province);
}
