import * as d3 from 'd3';

interface TaskSolution {
    start_time: number,
    execution_time: number,
    processor: number,
    name: string,
    color: string,
}

interface Data {
    throughput: number,
    tasks: TaskSolution[],
}

const margin = {
    top: 20,
    right: 20,
    bottom: 30,
    left: 40,
}

const zip = <A, B>(a: A[], b: B[]): [A, B][] => a.map((k, i) => [k, b[i]]);

function plot(
    content: d3.Selection<SVGGElement, unknown, HTMLElement, any>,
    x: d3.ScaleLinear<number, number, never>,
    y: d3.ScaleBand<number>,
    { throughput, tasks }: Data,
    width: number,
    height: number,
    offset: number,
    delta: number
) {

    content
        .selectAll(".execution")
        .data(tasks)
        .join(
            enter => enter
                .append("rect")
                .attr("class", "execution")
                .attr("height", y.bandwidth())
                .attr("stroke", "black")
                .attr("stroke-width", "1px")
                .attr("fill", t => t.color),
            update => update,
            exit => exit.remove()
        )
        .attr("x", (task) => x(task.start_time))
        .attr("width", (task) => x(task.start_time + task.execution_time) - x(task.start_time))
        .attr("y", (task) => y(task.processor));

}

export function main(id: string, data: Data) {
    const height = 150;
    let offset = 0;
    let delta = 1 / data.throughput;

    const svg = d3.select(id)
        .append("svg");

    const width = () => svg.node().getBoundingClientRect().width - margin.left - margin.right;

    svg
        .attr("viewBox", `0 0 ${width()} ${height}`)
        .attr("width", width())
        .attr("height", "height")
        .style("width", "100%");

    const content = svg.append("g");

    const x = d3.scaleLinear()
        .domain([offset, offset + delta])
        .range([margin.left, width() - margin.right]);

    const y = d3.scaleBand<number>()
        .domain([...new Set(data.tasks.map(t => t.processor))])
        .range([margin.top, height - margin.bottom]);

    const xAxis = d3.axisBottom(x);

    svg.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .call(xAxis);

    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y));

    const replot = () => {
        plot(content, x, y, data, width(), height, offset, delta);
    }

    const mousemove = (e: MouseEvent) => {
        let w = width() - margin.left - margin.right;
        let deltaX = e.movementX;
        offset -= deltaX * delta / w
        x.domain([offset, offset + delta])
        svg.selectAll<SVGGElement, unknown>(".x-axis").call(xAxis);
        replot()
    }
    const mouseup = () => {
        document.body.style.cursor = null;
        svg.node().style.cursor = "grab";
        window.removeEventListener("mouseup", mouseup)
        window.removeEventListener("mousemove", mousemove)
    }
    const mousedown = () => {
        document.body.style.cursor = "grabbing"
        svg.node().style.cursor = null;
        window.addEventListener("mouseup", mouseup)
        window.addEventListener("mousemove", mousemove)
    }
    svg.node().addEventListener("mousedown", mousedown)

    function wheel(e: WheelEvent) {
        replot();
    }
    svg.node().addEventListener("wheel", wheel)

    const resizeObserver = new ResizeObserver(() => {
        svg
            .attr("viewBox", [0, 0, width(), height])
            .attr("width", width);
        x.range([margin.left, width() - margin.right]);
        svg.selectAll<SVGGElement, unknown>(".x-axis").call(xAxis);
        replot();
    });
    resizeObserver.observe(svg.node());

    replot();
}