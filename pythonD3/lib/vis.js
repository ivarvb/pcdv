function plotClassification(slide, data, samples, dimered, datanor, visuali, classif){
  d3.select("#loadingId").style("display", "block");
  var url = "./pl/?tq=3&data="+data+"&samples="+samples+"&dimered="+dimered+"&datanor="+datanor+"&visuali="+visuali+"&classif="+classif;
  loadAjax('#classChart',url);
  //d3.tsv(url, function(dat) {

  d3.select("#loadingId").style("display", "none");
  //  console.log(dat);
  //});
}

function plotVisualization(slide, data, samples, dimered, datanor, visuali, classif){
  d3.select("#loadingId").style("display", "block");
  var url = "./pl/?tq=1&data="+data+"&samples="+samples+"&dimered="+dimered+"&datanor="+datanor+"&visuali="+visuali+"&classif="+classif;
  //var url = "./data/?data=1&samples=1&dimered=1&visuali=1&classif=1";
  //alert(url);
  d3.tsv(url, function(dat) {
      dat.forEach(function(d) {
          d.id = +d.id;
          d.x = +d.x;
          d.y = +d.y;
          d.c = +d.c;
      });
      /*var datar=[];
      for (var i = 0; i < data.length; i++) {
        var row = [];
        row.push(data[i].id);
        row.push(data[i].x);
        row.push(data[i].y);
        row.push(data[i].c);
        datar.push(row);
      }
      */
      d3.select("#loadingId").style("display", "none");
      console.log(dat);
      plotProjection(slide, dat);
  });


}

function plotProjection(slide, data){
  var margin = {top: 20, right: 20, bottom: 30, left: 40},
      width = slide - margin.left - margin.right,
      height = slide - margin.top - margin.bottom;

/*
 * value accessor - returns the value to encode for a given data object.
 * scale - maps value to a visual display encoding, such as a pixel position.
 * map function - maps from data value to display value
 * axis - sets up axis
 */

// setup x
var xValue = function(d) { return d.x;}, // data -> value
    xScale = d3.scale.linear().range([0, width]), // value -> display
    xMap = function(d) { return xScale(xValue(d));}, // data -> display
    xAxis = d3.svg.axis().scale(xScale).orient("bottom");

// setup y
var yValue = function(d) { return d.y;}, // data -> value
    yScale = d3.scale.linear().range([height, 0]), // value -> display
    yMap = function(d) { return yScale(yValue(d));}, // data -> display
    yAxis = d3.svg.axis().scale(yScale).orient("left");

// setup fill color
var cValue = function(d) { return d.c;},
    color = d3.scale.category10();

// add the graph canvas to the body of the webpage
d3.select("#chartsp").select("svg").remove();

var svg = d3.select("#chartsp")
    .append("svg")
    .attr("width", width + 100 + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// add the tooltip area to the webpage
var tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("display", 'block');

// load data


    //var data = [
    //          [0, 5, 20, 0], [1, 480, 90, 1], [2, 250, 50, 3], [3, 100, 33, 2], [4, 330, 95, 0],
    //        [5, 410, 12, 1], [6, 475, 44, 2], [7, 25, 67, 1], [8, 85, 21, 0], [9, 220, 88, 0]
    //        ];
    //var data = [

    // don't want dots overlapping axis, so add in buffer to data domain
    xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
    yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

    // x-axis
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis)
        .append("text")
        .attr("class", "label")
        .attr("x", width)
        .attr("y", -6);

    // y-axis
    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
        .append("text")
        .attr("class", "label")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71em");

/*
        .enter().append("rect")
        .attr("class", "dot")
        .attr("width", 3.5)
        .attr("height", 3.5)
        .attr("x", xMap)
        .attr("y", yMap)*/

    // draw dots
    svg.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("r", 3.5)
        .attr("cx", xMap)
        .attr("cy", yMap)
        .style("fill", function(d) { return color(cValue(d));})
        .on("mouseover", function(d) {
            tooltip.style("display", 'block');
            tooltip.html("<img src='./lib/img/py.jpeg' style='width:80px'>"+ d.id + "<br/> (" + xValue(d)
                + ", " + yValue(d) + ")")
                .style("left", (d3.event.pageX + 5) + "px")
                .style("top", (d3.event.pageY - 28) + "px");
        })
        .on("mouseout", function(d) {
            tooltip.style("display", 'none');
        });

    // draw legend
    var legend = svg.selectAll(".legend")
        .data(color.domain())
        .enter().append("g")
        .attr("class", "legend")
        .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

    // draw legend colored rectangles
    legend.append("rect")
        .attr("x", width + 18)
        .attr("width", 18)
        .attr("height", 18)
        .style("fill", color);

    // draw legend text
    legend.append("text")
        .attr("x", width + 40)
        .attr("y", 9)
        .attr("dy", ".35em")
        .text(function(d) { return d+"";});

    // opacity points
    d3.select("#range1").on("input", function () {
        svg.selectAll('circle')
        .style("opacity", d3.select("#range1").property("value")/100);
    });

}

function loadAjax(id, url){
  $.ajax({
      url: url,
      type: 'GET',
      dataType: 'html',
      success: function(html) {
          $(id).html(html);
      }
  });
}
