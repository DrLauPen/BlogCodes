package com.example.travels.service;

import com.example.travels.dao.PlaceDAO;
import com.example.travels.entity.Place;
import com.example.travels.entity.Province;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@Transactional
@Slf4j
public class PlaceServiceImpl implements PlaceService{

    @Autowired
    private PlaceDAO placeDAO;

    @Autowired
    private ProvinceSeriviceImpl provinceSerivice;

    @Override
    public void updatePlace(Place place) {
        //更新景点的信息
        placeDAO.update(place);
    }

    @Override
    public List<Place> findByProvinceIdPages(Integer page, Integer rows, String provinceId) {
        int start  = (page-1)*rows;
        return placeDAO.findByProvinceIdPages(start,rows,provinceId);
    }

    @Override
    public Integer findByProvinceIdCounts(String provinceId) {
        return placeDAO.findByProvinceIdCounts(provinceId);
    }

    @Override
    public Place findOnePlace(String id) {
        //查找对应的一个景点的信息
        return placeDAO.findOne(id);
    }

    @Override
    public void deletePlace(String id) {
        //对应的省份景点数目-1
        //首先查找到该景点
        Place place = placeDAO.findOne(id);
        //根据该景点的provinceid修改对应的省份的景点数目
        Province province = provinceSerivice.findOne(place.getProvinceid());
        province.setPlacecounts(province.getPlacecounts()-1);
        provinceSerivice.updateProvince(province);

        //删除对应的景点
        placeDAO.delete(id);
    }

    @Override
    public void save(Place place) {
        //保存对应的景点信息到数据库中
        placeDAO.save(place);
        //查询原始省份信息并修改placecounts为+1
        Province one = provinceSerivice.findOne(place.getProvinceid());
        one.setPlacecounts(one.getPlacecounts()+1);
        provinceSerivice.updateProvince(one);
    }
}
