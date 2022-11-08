<template>
  <q-layout id="background">
    <div style="float: left; margin-top: 0.2%; margin-left: 1%; width: 23%">
      <div class="column">
        <q-file v-model="file" label="Select an image (<=500KB)" accept="image/png,image/gif,image/jpeg" style="width: 100%; margin-left: 1%"/>
        <q-img
          :src="picUrl"
          style="max-height: 350px; margin-top: 1%"
        />
        <q-btn outline rounded label="Samples"  @click="show_samples = true" style="color: #03263A; margin-top: 1%; height: 20%"/>
        <q-btn outline rounded label="Color it!"  @click="get_color_open" style="color: #03263A; margin-top: 1%; height: 20%"/>
      </div>
      <q-dialog v-model="show_samples">
        <div class="q-pa-md" style="">
          <q-carousel
            swipeable
            animated
            v-model="slide"
            ref="carousel"
            infinite
          >
            <q-carousel-slide :name="1" img-src="../../public/samples/1.jpg" style="width: 400px"/>
            <q-carousel-slide :name="2" img-src="../../public/samples/2.jpg" style="width: 400px"/>
            <q-carousel-slide :name="3" img-src="../../public/samples/3.jpg" style="width: 400px"/>
            <q-carousel-slide :name="4" img-src="../../public/samples/4.jpg" style="width: 400px"/>
            <q-carousel-slide :name="5" img-src="../../public/samples/5.jpg" style="width: 400px"/>
            <q-carousel-slide :name="6" img-src="../../public/samples/6.jpg" style="width: 400px"/>
            <q-carousel-slide :name="7" img-src="../../public/samples/7.jpg" style="width: 400px"/>

            <template v-slot:control>

              <q-carousel-control
                position="bottom-right"
                :offset="[18, 18]"
                class="q-gutter-xs"
              >
                <q-btn
                  push round dense color="white" text-color="black" icon="eva-arrow-ios-back-outline"
                  @click="$refs.carousel.previous()"
                />
                <q-btn
                  push round dense color="white" text-color="black" icon="eva-arrow-ios-forward-outline"
                  @click="$refs.carousel.next()"
                />
                <q-btn
                  push round dense color="white" text-color="black" icon="eva-checkmark-outline"
                  @click="choose_sample"
                />
              </q-carousel-control>
            </template>
          </q-carousel>
        </div>
      </q-dialog>
      <div style="height: 1px; width: 100%; background: silver; margin-top: 5%; margin-bottom: 5%" />
      <div class="q-gutter-sm">
        <div style="font-size: medium; margin-top: 5%; font-weight: bold"> Chart Type </div>
        <q-radio style="font-size: medium" color="grey" v-model="shape" val="bar" label="Bar"/>
        <q-radio style="font-size: medium" color="grey" v-model="shape" val="line" label="Line"/>
        <q-radio style="font-size: medium" color="grey" v-model="shape" val="pie" label="Pie" />
        <q-radio style="font-size: medium" color="grey" v-model="shape" val="arealine" label="Area Line" />
        <q-radio style="font-size: medium" color="grey" v-model="shape" val="bubble" label="Line Scatter" />
      </div>
      <div style="height: 1px; width: 100%; background: silver; margin-top: 2%; margin-bottom: 3%" />
      <div style="font-size: medium; float: left; font-weight: bold"> Color </div>
      <div style="margin-top: 12%;" class="row">
        <div v-for="btn_id1 in btn_ids1" :key="btn_id1.id">
          <q-btn push :id="btn_id1.id" @click="copy(btn_id1.id)" label="#C0C0C0" style="background-color: #c0c0c0; width: 60px; margin-left: 10px;
        color: #ffffff; font-size: 15px; font-family: 'Adobe Garamond Pro'; margin-bottom: 5px; height: 25px;"/>
        </div>
      </div>
      <div style="margin-top: 2px" class="row">
        <div v-for="btn_id2 in btn_ids2" :key="btn_id2.id">
          <q-btn push :id="btn_id2.id" @click="copy(btn_id2.id)" label="#C0C0C0" style="background-color: #c0c0c0; width: 60px; margin-left: 10px;
        color: #ffffff; font-size: 15px; font-family: 'Adobe Garamond Pro'; margin-bottom: 5px; height: 25px;"/>
        </div>
      </div>
      <div style="margin-top: 2px" class="row">
        <div v-for="btn_id3 in btn_ids3" :key="btn_id3.id">
          <q-btn push :id="btn_id3.id" label="#C0C0C0" style="background-color: #c0c0c0; width: 74px; margin-left: 10px;
        color: #ffffff; font-size: 15px; font-family: 'Adobe Garamond Pro'; margin-bottom: 5px; height: 25px;"/>
        </div>
      </div>
      <div style="height: 1px; width: 100%; background: silver; margin-top: 2%; margin-bottom: 3%" />
      <div style="font-size: medium; float: left; font-weight: bold"> Background </div>
      <div style="margin-top: 12%;" class="row">
        <div v-for="btn_id4 in btn_ids4" :key="btn_id4.id">
          <q-btn push :id="btn_id4.id" @click="change_background_color(btn_id4.id)" label="#C0C0C0" style="background-color: #c0c0c0; width: 60px; margin-left: 10px;
        color: #ffffff; font-size: 15px; font-family: 'Adobe Garamond Pro'; margin-bottom: 5px; height: 25px;"/>
        </div>
      </div>
      <div style="height: 1px; width: 100%; background: silver; margin-top: 2%; margin-bottom: 3%" />
      <div style="margin-top: 2%">
        <div style="font-size: medium; float: left; font-weight: bold; margin-right: 20%"> Number of Color </div>
        <q-select v-model="colors_num" :options="colors_options" label="Standard" />
      </div>
    </div>
    <div style="float: right; margin-right: 4%; width: 70%">
      <q-toolbar class="bg-grey-3 text-grey-9 q-my-md shadow-2">
        <div style="font-family: 'Adobe Garamond Pro'; font-size: xx-large">
          ColorChart
        </div>

        <q-space />
        <div>
          <q-radio v-model="background_exist" val="true" label="Background" />
          <q-radio v-model="background_exist" val="false" label="Non-Background" />
        </div>
        <q-btn stretch flat label="background_color">
          <q-menu v-model="color_picker">
            <q-color v-model="background_color" dark class="my-picker" default-value="#ffffff"/>
          </q-menu>
        </q-btn>
        <q-separator dark vertical />
      </q-toolbar>
      <div id='chart' style="margin-top: 10%; margin-left: 10%; height: 550%; width: 80%"></div>
    </div>
  </q-layout>
</template>
<script>
import * as echarts from 'echarts'
import axios from 'axios'
import Vue from 'vue'
import VueClipboard from 'vue-clipboard2'

Vue.use(VueClipboard)

var chartdom
var myChart
// eslint-disable-next-line camelcase
var option_now

var colors = ['#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0']
// eslint-disable-next-line camelcase
var background_colors = ['#C0C0C0']

// eslint-disable-next-line camelcase
var data_line_2 = [[120, 132, 101, 134, 90, 230, 210], [220, 182, 191, 234, 290, 330, 310], [150, 232, 201, 154, 190, 330, 410],
  [270, 282, 251, 284, 340, 280, 270], [320, 306, 340, 371, 350, 380, 380], [350, 386, 370, 451, 450, 500, 510],
  [820, 932, 901, 934, 1290, 1330, 1320]]

// eslint-disable-next-line camelcase
var option_line_2 = {
  toolbox: {
    feature: {
      saveAsImage: {}
    }
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true
  },
  xAxis: [
    {
      type: 'category',
      boundaryGap: false,
      data: ['0', '1', '2', '3', '4', '5', '6'],
      show: false
    }
  ],
  yAxis: [
    {
      axisLabel: { show: false },
      splitLine: {
        show: false
      },
      type: 'value'
    }
  ],
  series: []
}
// eslint-disable-next-line camelcase
var data_bubble_1_hours = ['12a', '1a', '2a', '3a', '4a', '5a', '6a', '7a', '8a',
  '9a', '10a', '11a', '12p', '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p', '10p', '11p']
// eslint-disable-next-line camelcase
var data_bubble_1_days = ['Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday', 'Sunday']
// eslint-disable-next-line camelcase
var data_bubble_1 = [[0, 0, 5], [0, 1, 1], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 6, 0], [0, 7, 0],
  [0, 8, 0], [0, 9, 0], [0, 10, 0], [0, 11, 2], [0, 12, 4], [0, 13, 1], [0, 14, 1], [0, 15, 3],
  [0, 16, 4], [0, 17, 6], [0, 18, 4], [0, 19, 4], [0, 20, 3], [0, 21, 3], [0, 22, 2], [0, 23, 5],
  [1, 0, 7], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0], [1, 5, 0], [1, 6, 0], [1, 7, 0],
  [1, 8, 0], [1, 9, 0], [1, 10, 5], [1, 11, 2], [1, 12, 2], [1, 13, 6], [1, 14, 9], [1, 15, 11],
  [1, 16, 6], [1, 17, 7], [1, 18, 8], [1, 19, 12], [1, 20, 5], [1, 21, 5], [1, 22, 7], [1, 23, 2],
  [2, 0, 1], [2, 1, 1], [2, 2, 0], [2, 3, 0], [2, 4, 0], [2, 5, 0], [2, 6, 0], [2, 7, 0],
  [2, 8, 0], [2, 9, 0], [2, 10, 3], [2, 11, 2], [2, 12, 1], [2, 13, 9], [2, 14, 8], [2, 15, 10],
  [2, 16, 6], [2, 17, 5], [2, 18, 5], [2, 19, 5], [2, 20, 7], [2, 21, 4], [2, 22, 2], [2, 23, 4],
  [3, 0, 7], [3, 1, 3], [3, 2, 0], [3, 3, 0], [3, 4, 0], [3, 5, 0], [3, 6, 0], [3, 7, 0],
  [3, 8, 1], [3, 9, 0], [3, 10, 5], [3, 11, 4], [3, 12, 7], [3, 13, 14], [3, 14, 13], [3, 15, 12],
  [3, 16, 9], [3, 17, 5], [3, 18, 5], [3, 19, 10], [3, 20, 6], [3, 21, 4], [3, 22, 4], [3, 23, 1],
  [4, 0, 1], [4, 1, 3], [4, 2, 0], [4, 3, 0], [4, 4, 0], [4, 5, 1], [4, 6, 0], [4, 7, 0],
  [4, 8, 0], [4, 9, 2], [4, 10, 4], [4, 11, 4], [4, 12, 2], [4, 13, 4], [4, 14, 4], [4, 15, 14],
  [4, 16, 12], [4, 17, 1], [4, 18, 8], [4, 19, 5], [4, 20, 3], [4, 21, 7], [4, 22, 3], [4, 23, 0],
  [5, 0, 2], [5, 1, 1], [5, 2, 0], [5, 3, 3], [5, 4, 0], [5, 5, 0], [5, 6, 0], [5, 7, 0],
  [5, 8, 2], [5, 9, 0], [5, 10, 4], [5, 11, 1], [5, 12, 5], [5, 13, 10], [5, 14, 5], [5, 15, 7],
  [5, 16, 11], [5, 17, 6], [5, 18, 0], [5, 19, 5], [5, 20, 3], [5, 21, 4], [5, 22, 2], [5, 23, 0],
  [6, 0, 1], [6, 1, 0], [6, 2, 0], [6, 3, 0], [6, 4, 0], [6, 5, 0], [6, 6, 0], [6, 7, 0],
  [6, 8, 0], [6, 9, 0], [6, 10, 1], [6, 11, 0], [6, 12, 2], [6, 13, 1], [6, 14, 3], [6, 15, 4],
  [6, 16, 0], [6, 17, 0], [6, 18, 0], [6, 19, 0], [6, 20, 1], [6, 21, 2], [6, 22, 2], [6, 23, 6]]

// eslint-disable-next-line camelcase
var option_bubble_1 = {
  toolbox: {
    feature: {
      saveAsImage: {}
    }
  },
  tooltip: {
    position: 'top'
  },
  title: [],
  singleAxis: [],
  series: []
}

// eslint-disable-next-line camelcase
var data_line_1_5 = [[120, 132, 101, 134, 90, 230, 210], [220, 182, 191, 234, 290, 330, 310], [150, 232, 201, 154, 190, 330, 410],
  [320, 332, 301, 334, 390, 330, 320], [300, 532, 920, 1300, 1500, 1600, 1700]]
// eslint-disable-next-line camelcase
var data_line_1_6 = [[120, 132, 101, 134, 90, 230, 210], [220, 182, 191, 234, 290, 330, 310], [150, 232, 201, 154, 190, 330, 410],
  [320, 332, 301, 334, 390, 330, 320], [300, 400, 422, 566, 652, 800, 800], [300, 532, 920, 1300, 1500, 1600, 1700]]
// eslint-disable-next-line camelcase
var data_line_1_7 = [[120, 132, 101, 134, 90, 230, 210], [220, 182, 191, 234, 290, 330, 310], [150, 232, 201, 154, 190, 330, 410],
  [320, 332, 301, 334, 390, 330, 320], [300, 400, 422, 550, 550, 600, 620], [300, 300, 400, 500, 500, 550, 590], [300, 532, 920, 1300, 1500, 1800, 2300]]
// eslint-disable-next-line camelcase
var data_line_1_8 = [[120, 132, 101, 134, 90, 230, 210], [220, 182, 191, 234, 290, 330, 310], [150, 232, 201, 154, 190, 330, 410],
  [320, 332, 301, 334, 390, 330, 320], [300, 400, 422, 550, 550, 600, 620], [300, 300, 400, 500, 500, 550, 590],
  [300, 300, 400, 400, 400, 500, 560], [300, 532, 920, 1300, 1500, 1800, 2500]]
// eslint-disable-next-line camelcase
var data_line_1_9 = [[120, 132, 101, 134, 90, 230, 210], [220, 182, 191, 234, 290, 330, 310], [150, 232, 201, 154, 190, 330, 410],
  [320, 332, 301, 334, 390, 330, 320], [300, 400, 422, 550, 550, 600, 620], [300, 300, 400, 500, 500, 550, 590],
  [300, 300, 400, 400, 400, 500, 560], [300, 300, 400, 400, 400, 500, 500], [300, 532, 920, 1500, 1600, 2000, 2600]]
// eslint-disable-next-line camelcase
var data_line_1_10 = [[120, 132, 101, 134, 90, 230, 210], [220, 182, 191, 234, 290, 330, 310], [150, 232, 201, 154, 190, 330, 410],
  [320, 332, 301, 334, 390, 330, 320], [300, 400, 422, 550, 550, 600, 620], [300, 300, 400, 500, 500, 550, 590],
  [300, 300, 400, 400, 400, 500, 560], [300, 300, 400, 400, 400, 500, 500], [150, 232, 201, 154, 190, 330, 410], [300, 532, 920, 1500, 1600, 2200, 2800]]
// eslint-disable-next-line camelcase
var option_line_1 = {
  tooltip: {
    trigger: 'axis'
  },
  legend: {
    data: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    show: true
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true
  },
  toolbox: {
    feature: {
      saveAsImage: {}
    }
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: ['0', '1', '2', '3', '4', '5', '6'],
    show: false
  },
  yAxis: {
    type: 'value',
    axisLabel: { show: false },
    splitLine: {
      show: false
    }
  },
  series: []
}
// eslint-disable-next-line camelcase
var data_pie_1 = [1048, 835, 650, 580, 384, 300, 251]
// eslint-disable-next-line camelcase
var option_pie_1 = {
  legend: {
    orient: 'vertical',
    left: 'left'
  },
  toolbox: {
    feature: {
      saveAsImage: {}
    }
  },
  series: [
    {
      type: 'pie',
      radius: ['80%'],
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 10,
        borderColor: '#fff',
        borderWidth: 2
      },
      label: {
        show: false,
        position: 'center'
      },
      emphasis: {
        label: {
          show: false,
          fontSize: '40',
          fontWeight: 'bold'
        }
      },
      labelLine: {
        show: false
      },
      data: []
    }
  ]
}
// eslint-disable-next-line camelcase
var data_bar_1 = [[0, 0, 40, 25, 72, 92, 55, 0, 0, 0], [0, 0, 50, 72, 40, 92, 59, 25, 0, 0], [0, 46, 100, 72, 50, 30, 89, 40, 0, 0],
  [0, 46, 100, 72, 50, 30, 89, 40, 64, 0], [78, 46, 100, 72, 50, 30, 89, 40, 64, 0], [78, 46, 100, 72, 50, 30, 89, 40, 64, 23]]

// eslint-disable-next-line camelcase
var option_bar_1 = {
  legend: {
    orient: 'vertical',
    left: 'left'
  },
  xAxis: {
    show: false,
    type: 'category',
    data: ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10']
  },
  yAxis: {
    axisLabel: { show: false },
    type: 'value',
    splitLine: {
      show: false
    }
  },
  series: [{
    barCategoryGap: '10%',
    data: [],
    type: 'bar',
    label: {
      show: false
    },
    itemStyle: {
      emphasis: {
        barBorderRadius: 15
      },
      normal: {
        barBorderRadius: 15
      }
    }
  }
  ],
  toolbox: { show: true, feature: { saveAsImage: { show: true, excludeComponents: ['toolbox'], pixelRatio: 2, type: 'png' } } }
}

export default {
  watch: {
    background_exist (val, oldVal) {
      var that = this
      if (val === 'true') {
        option_now.backgroundColor = that.background_color
      } else {
        option_now.backgroundColor = 'rgba(128, 128, 128, 0)'
      }
      myChart.clear()
      myChart.setOption(option_now)
      // eslint-disable-next-line camelcase
    },
    file (val, oldVal) {
      var that = this
      if (val.size / 1024 <= 500) {
        that.change_pic(val, true)
      } else {
        alert('The image is more than 500KB!')
      }
    },
    shape (val, oldVal) {
      var that = this
      if (val === 'bar') {
        that.change_graph_bar1()
      } else if (val === 'line') {
        that.change_graph_line1()
      } else if (val === 'pie') {
        that.change_graph_pie1()
      } else if (val === 'arealine') {
        that.change_graph_line2()
      } else if (val === 'bubble') {
        that.change_graph_bubble1()
      }
    },
    colors_num (val, oldVal) {
      var legends = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
      var that = this
      chartdom = document.getElementById('chart')
      // eslint-disable-next-line camelcase
      if (option_now === option_bar_1) {
        option_now.series[0].data = []
        var ikk = 0
        for (var ik = 0; ik < 10; ik++) {
          if (data_bar_1[val - 5][ik] === 0) {
            option_now.series[0].data.push({
              value: data_bar_1[val - 5][ik],
              itemStyle: {
                color: '#ffffff'
              }
            })
          } else {
            option_now.series[0].data.push({
              value: data_bar_1[val - 5][ik],
              itemStyle: {
                color: colors[ikk]
              },
              name: legends[ikk]
            })
            ikk += 1
          }
        }
      }
      that.btn_ids3 = []
      that.btn_ids2 = [{ id: 'btn5' }]
      for (var j = 6; j <= val; j++) {
        if (j === 9) {
          break
        }
        that.btn_ids2.push({ id: ('btn' + j.toString()) })
      }
      for (var i = 9; i <= val; i++) {
        that.btn_ids3.push({ id: ('btn' + i.toString()) })
      }
      that.get_color_open()
    },
    background_color (val, oldVal) {
      var that = this
      option_now.backgroundColor = val
      that.background_color = val
      // if (val[0] === '#') {
      //   that.background_exist = 'true'
      // }
      chartdom = document.getElementById('chart')
      myChart = echarts.init(chartdom)
      myChart.setOption(option_now)
    }
  },
  data () {
    return {
      background_exist: 'false',
      colors_options: [],
      file: [],
      shape: 'bar',
      picUrl: '',
      colors_num: 5,
      dataset1: true,
      dataset2: false,
      btn_ids1: [{ id: 'btn1' }, { id: 'btn2' }, { id: 'btn3' }, { id: 'btn4' }],
      btn_ids2: [{ id: 'btn5' }],
      btn_ids3: [],
      btn_ids4: [{ id: 'bg_btn1' }],
      btn_ids5: [{ id: 'bg_btn5' }, { id: 'bg_btn6' }],
      btn_ids6: [{ id: 'bbg_btn1' }, { id: 'bbg_btn2' }],
      alert: false,
      background_color: '#ffffff',
      color_picker: false,
      show_samples: false,
      slide: 1
    }
  },
  mounted () {
    var that = this
    for (var ia = 5; ia < 8; ia++) {
      that.colors_options.push(ia)
    }
    var iik = 0
    var legends = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for (var ik = 0; ik < 10; ik++) {
      if (data_bar_1[that.colors_num - 5][ik] === 0) {
        option_bar_1.series[0].data.push({
          value: data_bar_1[that.colors_num - 5][ik],
          itemStyle: {
            color: '#ffffff'
          }
        })
      } else {
        option_bar_1.series[0].data.push({
          value: data_bar_1[that.colors_num - 5][ik],
          itemStyle: {
            color: colors[iik]
          },
          name: legends[iik]
        })
        iik += 1
      }
    }
    chartdom = document.getElementById('chart')
    myChart = echarts.init(chartdom)
    myChart.setOption(option_bar_1)
    // eslint-disable-next-line camelcase
    option_now = option_bar_1
  },
  methods: {
    // eslint-disable-next-line camelcase
    copy (button_id) {
      var index = parseInt(button_id[3] - 1)
      const copytext = colors[index]
      this.$copyText(copytext).then(res => {
        this.$message.success({ message: 'copy successed' })
      }).catch(err => {
        this.$messag1e.error({ message: err })
      })
    },

    change_btn_color (callback) {
      var that = this
      for (var i = 1; i <= that.colors_num; i++) {
        document.getElementById('btn' + i.toString()).style.backgroundColor = colors[i - 1]
        document.getElementById('btn' + i.toString()).textContent = colors[i - 1]
      }
    },
    // eslint-disable-next-line camelcase
    change_background_color (button_id) {
      var that = this
      that.background_color = background_colors[0]
    },
    choose_sample () {
      var that = this
      that.show_samples = false
      this.change_pic(null, false)
    },
    change_background_btn_color (callback) {
      var that = this
      that.background_color = background_colors[0]
      document.getElementById('bg_btn1').style.backgroundColor = background_colors[0]
      document.getElementById('bg_btn1').textContent = background_colors[0]
    },

    change_graph_color_data (option, callback) {
      var legends = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
      var that = this
      myChart.clear()
      // eslint-disable-next-line camelcase
      if (option === option_line_1) {
        option.legend.data = []
        option.series = []
        // eslint-disable-next-line camelcase
        for (var ij = 0; ij < that.colors_num; ij++) {
          option.legend.data.push('item' + ij.toString())
          // eslint-disable-next-line camelcase
          var Data_line = []
          if (that.colors_num === 5) {
            // eslint-disable-next-line camelcase
            Data_line = data_line_1_5
          } else if (that.colors_num === 6) {
            // eslint-disable-next-line camelcase
            Data_line = data_line_1_6
          } else if (that.colors_num === 7) {
            // eslint-disable-next-line camelcase
            Data_line = data_line_1_7
          } else if (that.colors_num === 8) {
            // eslint-disable-next-line camelcase
            Data_line = data_line_1_8
          } else if (that.colors_num === 9) {
            // eslint-disable-next-line camelcase
            Data_line = data_line_1_9
          } else {
            // eslint-disable-next-line camelcase
            Data_line = data_line_1_10
          }
          option.series.push({
            name: 'item' + ij.toString(),
            type: 'line',
            stack: 'all',
            data: Data_line[ij],
            color: colors[ij]
          })
        }
        // eslint-disable-next-line camelcase
      } else if (option === option_line_2) {
        option.series = []
        for (var io = 0; io < that.colors_num; io++) {
          // eslint-disable-next-line camelcase
          option.series.push({
            type: 'line',
            stack: 'all',
            areaStyle: {},
            emphasis: {
              focus: 'series'
            },
            data: data_line_2[io],
            color: colors[io]
          })
        }
        // eslint-disable-next-line camelcase
      } else if (option === option_bubble_1) {
        option.singleAxis = []
        option.series = []
        data_bubble_1_days.forEach(function (day, idx) {
          if (that.colors_num === 5) {
            option.singleAxis.push({
              left: 150,
              type: 'category',
              boundaryGap: false,
              data: data_bubble_1_hours,
              top: (idx * 100 / 5 + 5) + '%',
              height: (100 / 7 - 14) + '%',
              axisLabel: {
                interval: 2,
                show: false
              }
            })
          } else if (that.colors_num === 6) {
            option.singleAxis.push({
              left: 150,
              type: 'category',
              boundaryGap: false,
              data: data_bubble_1_hours,
              top: (idx * 100 / 6 + 5) + '%',
              height: (100 / 7 - 14) + '%',
              axisLabel: {
                interval: 2,
                show: false
              }
            })
          } else if (that.colors_num === 7) {
            option.singleAxis.push({
              left: 150,
              type: 'category',
              boundaryGap: false,
              data: data_bubble_1_hours,
              top: (idx * 100 / 7 + 5) + '%',
              height: (100 / 7 - 10) + '%',
              axisLabel: {
                interval: 2,
                show: false
              }
            })
          } else {
            console.log('number is not between 5 and 7')
          }
          // eslint-disable-next-line camelcase
          option.series.push({
            singleAxisIndex: idx,
            coordinateSystem: 'singleAxis',
            type: 'scatter',
            data: [],
            symbolSize: function (dataItem) {
              return dataItem[1] * 4
            },
            color: colors[idx]
          })
        })
        data_bubble_1.forEach(function (dataItem) {
          option.series[dataItem[0]].data.push([dataItem[1], dataItem[2]])
        })
        // eslint-disable-next-line camelcase
      } else if (option === option_bar_1) {
        option.series[0].data = []
        var ikk = 0
        for (var ik = 0; ik < 10; ik++) {
          if (data_bar_1[that.colors_num - 5][ik] === 0) {
            option.series[0].data.push({
              value: data_bar_1[that.colors_num - 5][ik],
              itemStyle: {
                color: '#000000'
              }
            })
          } else {
            option.series[0].data.push({
              value: data_bar_1[that.colors_num - 5][ik],
              itemStyle: {
                color: colors[ikk]
              }
            })
            ikk += 1
          }
        }
        var sorted = []
        for (var is = 0; is < data_bar_1[that.colors_num - 5].length; is++) {
          sorted[is] = data_bar_1[that.colors_num - 5][is]
        }
        sorted = sorted.sort(function (a, b) { return b - a })
        ikk = 0
        for (var iq = 0; iq < sorted.length; iq++) {
          if (sorted[iq] !== 0) {
            for (var ip = 0; ip < 10; ip++) {
              if (sorted[iq] === option.series[0].data[ip].value) {
                option.series[0].data[ip].itemStyle.color = colors[ikk]
                ikk += 1
                break
              }
            }
          }
        }
        // eslint-disable-next-line camelcase
      } else if (option === option_pie_1) {
        option.series[0].data = []
        for (var iu = 0; iu < that.colors_num; iu++) {
          option.series[0].data.push({
            value: data_pie_1[iu],
            itemStyle: {
              color: colors[iu]
            },
            name: legends[iu]
          })
        }
        // eslint-disable-next-line camelcase
        sorted = data_pie_1
        sorted = sorted.sort(function (a, b) { return b - a })
        ikk = 0
        for (var ir = 0; ir < sorted.length; ir++) {
          for (var it = 0; it < that.colors_num; it++) {
            if (sorted[ir] === option.series[0].data[it].value) {
              option.series[0].data[it].itemStyle.color = colors[ikk]
              ikk += 1
              break
            }
          }
        }
      } else {
        for (var j = 0; j < 7; j++) {
          option.series[0].data[j].itemStyle.color = colors[j]
        }
      }
      myChart.setOption(option)
      // eslint-disable-next-line camelcase
      option_now = option
      that.$q.loading.hide()
    },
    // eslint-disable-next-line camelcase
    change_pic (file, file_flag) {
      var that = this
      // eslint-disable-next-line camelcase
      if (file_flag) {
        var param = new FormData()
        param.append('file', file)
        axios.post('http://127.0.0.1:5000/get_pic', param, { responseType: 'arraybuffer' }).then(function (response) {
          that.picUrl = 'data:image/png;base64,' + that.arrayBufferToBase64(response.data)
          console.log(response.data)
        }).catch(function (error) {
          alert('Error ' + error)
        })
      } else {
        var index = 0
        if (that.$refs.carousel.$children[0].$options.propsData.imgSrc === undefined) {
          index = 1
        }
        console.log(that.$refs.carousel)
        axios.get('http://127.0.0.1:5000/get/get_sample', {
          params: {
            sample_index: that.$refs.carousel.$children[index].$options.propsData.imgSrc.slice(4)
          }
        }, { responseType: 'arraybuffer' }).then(function (response) {
          that.picUrl = 'data:image/png;base64,' + response.data
        }).catch(function (error) {
          alert('Error ' + error)
        })
      }
    },
    arrayBufferToBase64 (buffer) {
      var binary = ''
      var bytes = new Uint8Array(buffer)
      var len = bytes.byteLength
      for (var i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i])
      }
      return window.btoa(binary)
    },
    get_color_open () {
      var that = this
      if (that.file.length === 0) {
        alert('Please upload an image or select one from the samples.')
      } else {
        this.$q.loading.show({
          message: 'Generating……'
        })
        axios.get('http://127.0.0.1:5000/get/color_open', {
          params: {
            number: that.colors_num,
            bcg_flag: that.background_exist
          }
        }, { responseType: 'json' }).then(function (response) {
          colors = response.data.data.color_list

          // eslint-disable-next-line camelcase
          background_colors = [response.data.data.bcg]
          if (that.picUrl === '') {
            colors = []
            for (var k = 0; k < that.colors_num; k++) {
              colors.push('#C0C0C0')
            }
          }
          that.change_btn_color(function () {
          })
          that.change_background_btn_color(function () {
          })
          that.change_graph_color_data(option_now, function () {
          })
        }).catch(function (error) {
          alert('Error ' + error)
        })
      }
    },
    change_graph_bar1 () {
      var that = this
      that.change_graph_color_data(option_bar_1)
    },
    change_graph_line1 () {
      var that = this
      that.change_graph_color_data(option_line_1)
    },
    change_graph_pie1 () {
      var that = this
      that.change_graph_color_data(option_pie_1)
    },
    change_graph_line2 () {
      var that = this
      that.change_graph_color_data(option_line_2)
    },
    change_graph_bubble1 () {
      var that = this
      that.change_graph_color_data(option_bubble_1)
    }
  }
}
</script>
