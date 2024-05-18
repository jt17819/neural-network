const canvas = document.querySelector("canvas")
const toolBtns = document.querySelectorAll(".tool")
const clearCanvas = document.querySelector(".clear-canvas")
const sendImg = document.querySelector(".send-img")
const advanced = document.querySelector("#advanced")
let ctx = canvas.getContext("2d")

let isDrawing = false
let selectedOption = "my-nn"
let brushWidth = Math.floor(canvas.offsetWidth / 12)

const setCanvasBackground = () => {
    ctx.fillStyle = "#fff"
    ctx.fillRect(0, 0, canvas.width, canvas.height, )
    ctx.fillStyle = "#000"
}

window.addEventListener("load", () => {
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight
    setCanvasBackground()
})

const startDrawing = () => {
    isDrawing = true
    ctx.beginPath()
    ctx.lineWidth = brushWidth
    ctx.fillStyle = "#000"
}

const stopDrawing = () => {
    isDrawing = false
}

const drawing = (e) => {
    if(!isDrawing) return
    ctx.lineTo(e.offsetX, e.offsetY)
    ctx.stroke()
}

const postFetch = async (payload) => {
    const options = { 
        method: "POST",
        body: JSON.stringify({"payload": payload}),
        
        headers: new Headers({ "X-Api-Key": "AxZaVHIQOY1E9rkq2tiNu3p8IbDAi7FN9K667KSB", "Content-Type": "application/x-www-form-urlencoded"})
    }
    resp = await fetch("https://8g46y1f790.execute-api.eu-west-2.amazonaws.com/default/neural-network-demo", options)
    data = await resp.json()
    return data    
}

const showResult = (resp) => {
    const output = document.querySelector(".output")
    const container = document.createElement("div")
    while (output.hasChildNodes()) {
        output.removeChild(output.firstChild)
    }
    console.log(typeof(resp["raw"]))
    console.log(resp["raw"][0][1])
    if (advanced.checked) {
        const list = document.createElement("ul")
        for (let i = 0; i < 10; i++) {
            const item = document.createElement("li")
            item.textContent = `${i}: ${resp["raw"][0][i].toFixed(4)}%`
            if (i == resp["ans"]) {
                item.classList.add("answer")
            }
            console.log(resp["raw"][0][i])
            list.appendChild(item)
        }
        container.appendChild(list)
        output.appendChild(container)
    } else {
        const res = document.createElement("h3")
        res.textContent = resp["ans"]
        container.appendChild(res)
        output.appendChild(container)
    }
    const result = document.querySelector(".result.invisible")
    result.classList.remove("invisible")
    result.classList.add("visible")
}

toolBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelector(".options .active").classList.remove("active")
        btn.classList.add("active")
        selectedOption = btn.id
        console.log(selectedOption)
    })
})

clearCanvas.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    setCanvasBackground()
    document.querySelector(".result").classList.remove("visible")
    document.querySelector(".result").classList.add("invisible")
})

sendImg.addEventListener("click", async () => {
    // const link = document.createElement("a")
    // console.log(options)
    // link.download = `${Date.now()}.jpg`
    // link.href = canvas.toDataURL("image/jpg")
    // console.log(link.href)
    // link.click()
    prediction = await postFetch(canvas.toDataURL("image/jpg"))
    console.log(prediction)
    showResult(prediction)

})

canvas.addEventListener("mousedown", startDrawing)
canvas.addEventListener("mouseup", stopDrawing)
canvas.addEventListener("mousemove", drawing)
