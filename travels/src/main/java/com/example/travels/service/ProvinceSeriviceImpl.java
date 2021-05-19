package com.example.travels.service;

import com.example.travels.dao.ProvinceDAO;
import com.example.travels.entity.Province;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@Transactional
public class ProvinceSeriviceImpl implements ProvinceService{

    @Autowired
    private ProvinceDAO provinceDAO;

    @Override
    public List<Province> findByPages(Integer page, Integer rows) {
        int start = (page-1)*rows;
        return provinceDAO.findByPages(start,rows);
    }

    @Override
    public Integer findTotals() {
        return provinceDAO.findTotals();
    }

    @Override
    public void saveProvince(Province province) {
        //保存对应的省份到数据库中
        province.setPlacecounts(0);
        provinceDAO.save(province);
    }

    @Override
    public void deleteProvince(String id) {
        //删除对应的省份,有待改进，如果有对应的景点的时候，不能删除对应的省份.
        provinceDAO.delete(id);
    }

    @Override
    public Province findOne(String id) {
        //查询对应的单个省份
        return provinceDAO.findOne(id);
    }

    @Override
    public void updateProvince(Province province) {
        //更新对应的省份信息
        provinceDAO.update(province);
    }
}
