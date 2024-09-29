import styled from "styled-components"
import { useRef, useState, useEffect } from "react"

import star from './img/star.png'
import upload from './img/upload_ready.png'
import trash from './img/trash.png'
import play_white from './img/play_white.png'
import play_grey from './img/play_grey.png'
import prev from './img/prev.png'
import next from './img/next.png'
import play from './img/play.png'

import JSZip from "jszip";

import './App.css'
import { Tab, TabList, TabPanel, TabPanels, Tabs } from "@chakra-ui/react"


const Warper = styled.div`
    width: 100%;
    min-height: 100dvh;
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: #0D0C16;
    box-sizing: border-box;
`

const Container = styled.div`
    width: 100%;
    max-width: 1336px;
`

const HeaderContainer = styled.div`
    height: 64px;
    border-bottom: 1px solid #000000;
    box-shadow: 0px 0px 10px 0px rgba(48, 143, 255, 0.25);
    width: 100%;
    min-width: 100dvw;
    display: flex;
    justify-content: center;
`

const Header = styled.div`
    width: 100%;
    display: flex;
    justify-content: space-between;
    padding: 0px 35px;
    align-items: center;
    max-width: 1336px;
`

const LeftLogo = styled.div`
    display: flex;
    align-items: center;
    column-gap: 40px;
`

const StarLogo = styled.img`
    width: 19px;
    height: 18.98px;
`

const HeaderTitle = styled.span`
    font-family: 'Inter';
    font-weight: 900;
    font-size: 19px;
    color: #ffffff;
`

const HeaderLogoName = styled.span`
    font-family: 'Inter';
    font-weight: 900;
    font-size: 20px;
    color: #ffffff;
`

const MainContent = styled.div`
    display: flex;
    column-gap: 34px;
    justify-content: center;
    padding-top: 48px;
`

const LeftTable = styled.div`
    display: flex;
    flex-direction: column;
`

const Upload = styled.div`
    width: 260px;
    height: 80px;
    border-radius: 5px;
    display: flex;
    justify-content: center;
    align-items: center;
    column-gap: 12px;
    position: relative;
    background-color: #0D0C16;
    box-shadow: 0px 0px 10px 0px rgba(151, 153, 190, 0.25);
`

const UploadIcon = styled.img`
    height: 15px;
    width: auto;
`

const UploadText = styled.span`
    font-family: 'Inter';
    font-weight: 900;
    font-size: 13px;
    color: #ffffff;
`

const UploadTrash = styled.img`
    position: absolute;
    right: 8px;
    bottom: 6px;
    height: 13px;
    width: auto;
    cursor: pointer;
`

const UploadButton = styled.button`
    width: 260px;
    height: 46px;
    border-radius: 5px;
    border: none;
    background-color: #073dfd;
    font-family: 'Inter';
    font-weight: 900;
    font-size: 15px;
    color: #ffffff;
    margin-top: 17px;
    cursor: pointer;
    transition: all .3s ease-in-out;

    &:hover {
        box-shadow: 0px 0px 10px 3px rgba(7, 61, 253, 0.5);
    }
`

const ListClips = styled.div`
    display: flex;
    flex-direction: column;
    row-gap: 15px;
    padding: 19px 20px 25px 21px;
    width: 260px;
    height: 419px;
    border-radius: 5px;
    box-shadow: 0px 0px 7px 0px rgba(51, 182, 255, 0.3);
    align-items: center;
    overflow-y: auto;
    overflow-x: hidden;
    margin-top: 23px;
    scrollbar-gutter: stable;

    &::-webkit-scrollbar {
        background-color: #0D0C16;
        width: 2px;
    }

    &::-webkit-scrollbar-thumb {
        background-color: #242136;
    }
`

const ClipsInfo = styled.div`
    width: 205px;
    min-height: 50px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
`

const ClipHead = styled.div`
    display: flex;
    align-items: center;
    justify-content: space-between;
`

const ClipTitle = styled.span<{ $select?: boolean }>`
    font-family: 'Inter';
    font-weight: 900;
    font-size: 13px;
    color: ${props => props.$select ? '#ffffff': 'rgba(151, 153, 190, 0.64)'};
`

const ClipPlay = styled.img`
    width: 12px;
    height: 15px;
    cursor: pointer;
`

const ClipFooter = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
`

const ClipMetric = styled.span`
    font-family: 'Inter';
    font-weight: 900;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.37);
`

const ClipTime = styled.span`
    font-family: 'Inter';
    font-weight: 900;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.37);
`

const CenterTable = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    row-gap: 35px;
`

const VideoPlayerBlock = styled.div`
    display: flex;
    column-gap: 20px;
    align-items: center;
`

const VideoPlayerButton = styled.img`
    cursor: pointer;
    width: 28px;
    height: 28px;
`

const VideoBlock = styled.div`
    width: 270px;
    height: 480px;
    position: relative;
    border-radius: 3px;
    overflow: hidden;
    cursor: pointer;
`

const StyledVideo = styled.video`
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 3px;
`

const PlayButton = styled.img`
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translateX(-50%);
    width: 24px;
    height: 30px;
`

const ControlPanelBlock = styled.div`
    display: flex;
    flex-direction: column;
    width: 366px;
    row-gap: 17px;
`

const ProgressBarWrapper = styled.div`
    width: 100%;
    height: 39px;
    background-color: rgba(255, 255, 255, 0.1);
    position: relative;
    cursor: pointer;
    border-radius: 5px;
`

const ProgressBar = styled.div<{ width: number }>`
    width: ${props => props.width}%;
    height: 100%;
    background-color: #4BC400;
`

const ProgressHandle = styled.div<{ left: number }>`
    position: absolute;
    left: ${props => props.left}%;
    top: 0px;
    width: 1px;
    height: 39px;
    background-color: #4BC400;
    cursor: pointer;
    transform: translateX(-50%);
`

const RightTable = styled.div`
    width: 587px;
    height: 583px;
    border-radius: 5px;
    border: 1px solid rgba(48, 143, 255, 0.25);
    position: relative;

    & .css-fgp5ep {
        border-bottom: 0px solid;
        transition: all .3 ease-in-out;
    }

    & .css-edw2yw[aria-selected=false], .css-edw2yw[data-selected] {
        transition: all .3 ease-in-out;
        border-bottom: 1px solid rgba(48, 143, 255, 0.25) !important;
    }

    & .css-edw2yw[aria-selected=true], .css-edw2yw[data-selected] {
        transition: all .3 ease-in-out;
        border-bottom: none !important;
    }

    & .css-a5mhaz {
        padding-top: 30px;
        padding-left: 22px;
        display: flex;
        flex-direction: column;
    }
`

const TabsText = styled.span`
    font-family: 'Inter';
    font-weight: 900;
    font-size: 15px;
    color: #ffffff;
`

const ClipName = styled.span`
    font-family: 'Inter';
    font-weight: 900;
    font-size: 20px;
    color: #ffffff;
`

const ClipDescriptionTitle = styled.span`
    margin-top: 20px;
    font-family: 'Inter';
    font-weight: 900;
    font-size: 16px;
    color: #ffffff;
`

const ClipDescription = styled.span`
    margin-top: 11px;
    font-family: 'Inter';
    font-weight: 900;
    font-size: 13px;
    color: #ffffff;
`

const ClipHesh = styled.span`
    margin-top: 20px;
    font-family: 'Inter';
    font-weight: 900;
    font-size: 13px;
    color: #ffffff;
`

const ClipDownload = styled.button`
    border: none;
    border-radius: 5px;
    background-color: #073DFD;
    width: 130px;
    height: 39px;
    font-family: 'Inter';
    font-weight: 900;
    font-size: 15px;
    color: #ffffff;
    position: absolute;
    bottom: 25px;
    right: 18px;
`

const ClipEditBlock = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
`

const ClipEditTitle = styled.span`
    font-family: 'Inter';
    font-weight: 900;
    font-size: 10px;
    color: #ffffff;
`

const ClipEditButtons = styled.div`
    display: flex;
    align-items: center;
    column-gap: 20px;
`

const ClipEditCut = styled.button`
    background-color: #0D0C16;
    border: none;
    border-radius: 5px;
    box-shadow: 0px 0px 10px 0px rgba(151, 153, 190, 0.25);
    padding: 8px;
    color: #ffffff;
    font-family: 'Inter';
    font-weight: 900;
    font-size: 10px;
`

function App(){
    const videoRef = useRef<HTMLVideoElement>(null)
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [selectedClip, setSelectedClip] = useState<number | null>(null);
    const [videoClips, setVideoClips] = useState<any[]>([]);
    const [videoSrc, setVideoSrc] = useState<string | null>(null)
    const [fileName, setFileName] = useState<string>("Загрузить файл")
    const [isPlaying, setIsPlaying] = useState(false)
    const progressBarRef = useRef<HTMLDivElement>(null);
    const [progress, setProgress] = useState(0);
    const [dragging, setDragging] = useState(false);
    const [isMouseDown, setIsMouseDown] = useState(false);

    const togglePlayPause = () => {
    const video = videoRef.current
    if (video) {
        if (isPlaying) {
            video.pause()
        } else {
            video.play()
        }
        setIsPlaying(!isPlaying)
        }
    }

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
          setSelectedFile(file);
          setFileName(file.name);
          const videoURL = URL.createObjectURL(file);
          setVideoSrc(videoURL);
          uploadFile(file); // Отправка на бэкенд
        }
      };

      const uploadFile = async (file: File) => {
        const formData = new FormData();
        formData.append("file", file);
    
        try {
            const response = await fetch("http://127.0.0.1:8000/video/upload/", {
                method: "POST",
                body: formData,
            });
    
            if (response.ok) {
                const zipBlob = await response.blob(); // Get the response as a Blob
                const zip = await JSZip.loadAsync(zipBlob); // Load the zip file
                const clips: any[] = [];
    
                // Iterate over the zip file's files
                for (const filename of Object.keys(zip.files)) {
                    const fileData = await zip.files[filename].async("blob");
                    const videoURL = URL.createObjectURL(fileData);
                    clips.push({ title: filename, url: videoURL, duration: "00:00" }); // Add other metadata as needed
                }
    
                setVideoClips(clips); // Update the video clips state
            } else {
                console.error("Ошибка загрузки файла");
            }
        } catch (error) {
            console.error("Ошибка при отправке запроса", error);
        }
    };

    const updateProgress = () => {
        const video = videoRef.current;
        if (video && !dragging) {
            const percent = (video.currentTime / video.duration) * 100;
            setProgress(percent);
        }
    };

    const handleMouseDown = () => {
        setDragging(true);
        setIsMouseDown(true);
    };

    const handleMouseMove = (e: MouseEvent) => {
        if (isMouseDown) {
            const progressBar = progressBarRef.current; 
            if (progressBar) {
                const rect = progressBar.getBoundingClientRect();
                const offsetX = e.clientX - rect.left;
                const newProgress = Math.max(0, Math.min((offsetX / rect.width) * 100, 100));
                setProgress(newProgress);
                const video = videoRef.current;
                if (video) {
                    video.currentTime = (newProgress / 100) * video.duration;
                }
            }
        }
    };
    
    const handleMouseUp = () => {
        if (dragging) {
            setDragging(false);
            setIsMouseDown(false);
        }
    };

    const handleMouseLeave = () => {
        if (dragging) {
            handleMouseUp();
        }
    };

    const handleProgressBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
        const progressBar = progressBarRef.current;
        if (progressBar) {
            const rect = progressBar.getBoundingClientRect();
            const offsetX = e.clientX - rect.left;
            const newProgress = Math.max(0, Math.min((offsetX / rect.width) * 100, 100));
            setProgress(newProgress);
    
            const video = videoRef.current;
            if (video) {
                video.currentTime = (newProgress / 100) * video.duration;
            }
        }
    };

    const handleClipClick = (index: number) => {
        setSelectedClip(index);
        setVideoSrc(videoClips[index].url); // Adjust according to your video object structure
        setIsPlaying(true);
    };
    
    useEffect(() => {
        if (isMouseDown) {
            document.addEventListener("mousemove", handleMouseMove);
            document.addEventListener("mouseup", handleMouseUp);
        } else {
            document.removeEventListener("mousemove", handleMouseMove);
            document.removeEventListener("mouseup", handleMouseUp);
        }
        return () => {
            document.removeEventListener("mousemove", handleMouseMove);
            document.removeEventListener("mouseup", handleMouseUp);
        };
    }, [isMouseDown]);

    useEffect(() => {
        const video = videoRef.current;
        if (video) {
            video.addEventListener("timeupdate", updateProgress);
        }
        return () => {
            if (video) {
                video.removeEventListener("timeupdate", updateProgress);
            }
        };
    }, []);

    return (
        <Warper>
            <link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap" rel="stylesheet"></link>
            <HeaderContainer>
                <Header>
                        <LeftLogo>
                            <StarLogo src={star} />
                            <HeaderTitle>Генерация виральных клипов</HeaderTitle>
                        </LeftLogo>
                        <HeaderLogoName>GhostCoders</HeaderLogoName>
                </Header>
            </HeaderContainer>
            <Container>
                <MainContent>
                    <LeftTable>
                        <Upload onClick={() => document.getElementById('file-upload')?.click()}>
                        <input
                            type="file"
                            accept="video/*"
                            onChange={handleFileChange}
                            style={{ display: 'none' }}
                            id="file-upload"
                        />
                        <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', columnGap: '12px' }}></label>
                            <UploadIcon src={upload} />
                            <UploadText>{fileName}</UploadText>
                            <UploadTrash src={trash} onClick={() => setFileName("Загрузить файл")} />
                        </Upload>
                        <UploadButton>Сгенерировать</UploadButton>
                        <ListClips>
                        {videoClips.map((clip, index) => (
                            <ClipsInfo key={index} onClick={() => handleClipClick(index)}>
                                <ClipHead>
                                    <ClipTitle $select={selectedClip === index}>{clip.title}</ClipTitle>
                                    <ClipPlay src={selectedClip === index ? play_white : play_grey} />
                                </ClipHead>
                                <ClipFooter>
                                    <ClipMetric>{clip.metric}</ClipMetric>
                                    <ClipTime>{clip.duration}</ClipTime>
                                </ClipFooter>
                            </ClipsInfo>
                        ))}
                        </ListClips>
                    </LeftTable>
                    <CenterTable>
                        <VideoPlayerBlock>
                            <VideoPlayerButton src={prev} />
                            <VideoBlock onClick={togglePlayPause}>
                            {videoSrc ? (
                                <StyledVideo ref={videoRef} src={videoSrc} />
                                ) : (
                                <StyledVideo ref={videoRef}>
                                    <source src="" />
                                </StyledVideo>
                                )}
                                {!isPlaying && <PlayButton src={play} />}
                            </VideoBlock>
                            <VideoPlayerButton src={next} />
                        </VideoPlayerBlock>
                        <ControlPanelBlock>
                            <ProgressBarWrapper ref={progressBarRef} onMouseDown={handleMouseDown} onMouseLeave={handleMouseLeave} onClick={handleProgressBarClick}>
                                <ProgressHandle left={progress} />
                                {/* <ProgressBar width={progress} /> */}
                            </ProgressBarWrapper>
                        </ControlPanelBlock>
                    </CenterTable>
                    <RightTable>
                        <Tabs isFitted variant='enclosed'>
                            <TabList>
                                <Tab>
                                    <TabsText>Описание</TabsText>
                                </Tab>
                                <Tab>
                                    <TabsText>Редактор</TabsText>
                                </Tab>
                            </TabList>
                            <TabPanels>
                                <TabPanel>
                                    <ClipName>Наименование: какое-то название</ClipName>
                                    <ClipDescriptionTitle>Описание:</ClipDescriptionTitle>
                                    <ClipDescription>Перед вами представлено очень интересное видео. Оно сочетает в себе все и динамику и интересный контент и ну и еще очень оченьочень очень очень очень очень очень очень оченьочень очень очень очень очень очень очень оченьочень очень очень очень очень очень очень оченьочень очень очень очень очень очень очень оченьочень очень очень очень очень очень очень оченьочень очень очень очень очень очень очень оченьочень очень очень очень очень очень много всего можно сказать о нем., а что именно?</ClipDescription>
                                    <ClipHesh>Хештеги: #что-то1 #что-то2</ClipHesh>
                                </TabPanel>
                                <TabPanel>
                                    <ClipEditBlock>
                                        
                                    </ClipEditBlock>
                                </TabPanel>
                            </TabPanels>
                        </Tabs>
                        <ClipDownload>Скачать</ClipDownload>
                    </RightTable>
                </MainContent>
            </Container>
        </Warper>
    )
}

export default App